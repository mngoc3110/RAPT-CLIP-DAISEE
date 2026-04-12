from torch import nn
import torch.nn.functional as F
from models.Temporal_Model import *
from models.Prompt_Learner import *
from models.Text import class_descriptor_5_only_face
from models.Adapter import Adapter
from clip import clip
import copy
import itertools


class CosineClassifier(nn.Module):
    """τ-normalized cosine classifier — prevents mode collapse.
    
    Normalizes both features and class prototypes to unit sphere.
    Output = tau * cosine_similarity(features, prototypes)
    
    Why this prevents collapse:
    - All classes equidistant from origin on unit sphere
    - Random prototypes → diverse initial predictions guaranteed
    - Learnable tau controls confidence magnitude
    
    Reference: Kang et al., "Decoupling Representation and Classifier" (ICLR 2020)
    """
    def __init__(self, in_features, num_classes, tau_init=16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.tau = nn.Parameter(torch.tensor(float(tau_init)))
    
    def forward(self, x):
        x_norm = F.normalize(x.float(), dim=-1)
        w_norm = F.normalize(self.weight, dim=-1)
        return self.tau * (x_norm @ w_norm.t())

class GenerateModel(nn.Module):
    def __init__(self, input_text, clip_model, args):
        super().__init__()
        self.args = args
        
        self.is_ensemble = any(isinstance(i, list) for i in input_text)
        
        if self.is_ensemble:
            self.num_classes = len(input_text)
            self.num_prompts_per_class = len(input_text[0])
            self.input_text = list(itertools.chain.from_iterable(input_text))
            print(f"=> Using Prompt Ensembling with {self.num_prompts_per_class} prompts per class.")
        else:
            self.input_text = input_text

        self.prompt_learner = PromptLearner(self.input_text, clip_model, args)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.dtype = clip_model.dtype
        self.image_encoder = clip_model.visual

        # For EAA
        self.face_adapter = Adapter(c_in=512, reduction=4)

        # For MI Loss
        if args.dataset == "RAER":
            hand_crafted_prompts = class_descriptor_5_only_face
        elif args.dataset == "CK+":
            from models.Text import class_descriptor_ckplus
            hand_crafted_prompts = class_descriptor_ckplus
        elif args.dataset == "DAiSEE":
            from models.Text import class_descriptor_daisee
            hand_crafted_prompts = class_descriptor_daisee
        elif args.dataset == "DAiSEE4Level":
            from models.Text import class_descriptor_daisee_4level
            hand_crafted_prompts = class_descriptor_daisee_4level
        elif args.dataset == "DAiSEEBinary":
            from models.Text import class_descriptor_daisee_binary
            hand_crafted_prompts = class_descriptor_daisee_binary
        elif args.dataset == "DAiSEE4Discrete":
            from models.Text import class_descriptor_daisee4
            hand_crafted_prompts = class_descriptor_daisee4
        elif args.dataset in ("CAER", "CAER-S"):
            from models.Text import class_descriptor_caer
            hand_crafted_prompts = class_descriptor_caer
        elif args.dataset == "StudentEngagement":
            from models.Text import class_descriptor_student_engagement
            hand_crafted_prompts = class_descriptor_student_engagement
        elif args.dataset == "StudentEngagement6":
            from models.Text import class_descriptor_student_engagement_6
            hand_crafted_prompts = class_descriptor_student_engagement_6
        else:
            from models.Text import class_descriptor_7_only_face
            hand_crafted_prompts = class_descriptor_7_only_face
        self.tokenized_hand_crafted_prompts = torch.cat([clip.tokenize(p) for p in hand_crafted_prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(self.tokenized_hand_crafted_prompts).type(self.dtype)
        self.register_buffer("hand_crafted_prompt_embeddings", embedding)


        self.temporal_net = Temporal_Transformer_AttnPool(num_patches=args.num_segments,
                                                     input_dim=512,
                                                     depth=args.temporal_layers,
                                                     heads=8,
                                                     mlp_dim=1024,
                                                     dim_head=64)
        
        self.temporal_net_body = Temporal_Transformer_AttnPool(num_patches=args.num_segments,
                                                     input_dim=512,
                                                     depth=args.temporal_layers,
                                                     heads=8,
                                                     mlp_dim=1024,
                                                     dim_head=64)
        self.clip_model_ = clip_model
        self.project_fc = nn.Linear(1024, 512)
        
        # Gaze MLP Fusion Branch
        self.gaze_mlp = nn.Sequential(
            nn.Linear(3, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 512),
            nn.LayerNorm(512)
        )
        self.alpha_gaze = nn.Parameter(torch.tensor(0.0))

        # Cosine Classifier Head (bypasses CLIP text similarity)
        self.use_classifier_head = getattr(args, 'use_classifier_head', False)
        if self.use_classifier_head:
            num_cls = self.num_classes if self.is_ensemble else len(input_text)
            self.classifier_head = CosineClassifier(512, num_cls, tau_init=12.0)
            print(f"=> Using COSINE CLASSIFIER ({num_cls} classes, tau_init=12.0)")

        # MoCo Initialization
        if hasattr(args, 'use_moco') and args.use_moco:
            print("=> Initializing MoCoRank...")
            self.moco_dim = 512
            self.moco_k = args.moco_k
            self.moco_m = args.moco_m
            self.moco_t = args.moco_t

            # Create momentum encoders
            self.image_encoder_m = copy.deepcopy(self.image_encoder)
            self.face_adapter_m = copy.deepcopy(self.face_adapter)
            self.temporal_net_m = copy.deepcopy(self.temporal_net)
            self.temporal_net_body_m = copy.deepcopy(self.temporal_net_body)
            self.project_fc_m = copy.deepcopy(self.project_fc)
            self.gaze_mlp_m = copy.deepcopy(self.gaze_mlp)

            # Freeze momentum encoders
            for param in self.image_encoder_m.parameters(): param.requires_grad = False
            for param in self.face_adapter_m.parameters(): param.requires_grad = False
            for param in self.temporal_net_m.parameters(): param.requires_grad = False
            for param in self.temporal_net_body_m.parameters(): param.requires_grad = False
            for param in self.project_fc_m.parameters(): param.requires_grad = False
            for param in self.gaze_mlp_m.parameters(): param.requires_grad = False

            # Create queue
            self.register_buffer("queue", torch.randn(self.moco_dim, self.moco_k))
            self.queue = nn.functional.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.image_encoder.parameters(), self.image_encoder_m.parameters()):
            param_k.data = param_k.data * self.moco_m + param_q.data * (1. - self.moco_m)
        for param_q, param_k in zip(self.face_adapter.parameters(), self.face_adapter_m.parameters()):
            param_k.data = param_k.data * self.moco_m + param_q.data * (1. - self.moco_m)
        for param_q, param_k in zip(self.temporal_net.parameters(), self.temporal_net_m.parameters()):
            param_k.data = param_k.data * self.moco_m + param_q.data * (1. - self.moco_m)
        for param_q, param_k in zip(self.temporal_net_body.parameters(), self.temporal_net_body_m.parameters()):
            param_k.data = param_k.data * self.moco_m + param_q.data * (1. - self.moco_m)
        for param_q, param_k in zip(self.project_fc.parameters(), self.project_fc_m.parameters()):
            param_k.data = param_k.data * self.moco_m + param_q.data * (1. - self.moco_m)
        for param_q, param_k in zip(self.gaze_mlp.parameters(), self.gaze_mlp_m.parameters()):
            param_k.data = param_k.data * self.moco_m + param_q.data * (1. - self.moco_m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys) # Removed distributed gather for single GPU simplicity

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size > self.moco_k: # Handle wrap-around if batch size > remaining space
             batch_size = self.moco_k - ptr # truncate to fit
             keys = keys[:batch_size]
        
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.moco_k  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def forward_momentum(self, image_face, image_body, gaze_features=None):
        # Face Part
        n, t, c, h, w = image_face.shape
        image_face = image_face.contiguous().view(-1, c, h, w)
        image_face_features = self.image_encoder_m(image_face.type(self.dtype))
        image_face_features = self.face_adapter_m(image_face_features)
        image_face_features = image_face_features.contiguous().view(n, t, -1)
        video_face_features = self.temporal_net_m(image_face_features)
        
        # Body Part
        n, t, c, h, w = image_body.shape
        image_body = image_body.contiguous().view(-1, c, h, w)
        image_body_features = self.image_encoder_m(image_body.type(self.dtype))
        image_body_features = image_body_features.contiguous().view(n, t, -1)
        video_body_features = self.temporal_net_body_m(image_body_features)

        # Concatenate and Project
        video_features = torch.cat((video_face_features, video_body_features), dim=-1)
        video_features = self.project_fc_m(video_features)
        
        # Fuse Gaze Features
        if gaze_features is not None:
            gaze_avg = gaze_features.mean(dim=1)
            gaze_encoded = self.gaze_mlp_m(gaze_avg.type(self.dtype))
            video_features = video_features + self.alpha_gaze * gaze_encoded
            
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        return video_features
        
    def forward(self, image_face, image_body, gaze_features=None):
        ################# Visual Part #################
        # Face Part
        n, t, c, h, w = image_face.shape
        image_face_reshaped = image_face.contiguous().view(-1, c, h, w)
        image_face_features = self.image_encoder(image_face_reshaped.type(self.dtype))
        image_face_features = self.face_adapter(image_face_features) # Apply EAA
        image_face_features = image_face_features.contiguous().view(n, t, -1)
        video_face_features = self.temporal_net(image_face_features)  # (4*512)
        
        # Body Part
        n, t, c, h, w = image_body.shape
        image_body_reshaped = image_body.contiguous().view(-1, c, h, w)
        image_body_features = self.image_encoder(image_body_reshaped.type(self.dtype))
        image_body_features = image_body_features.contiguous().view(n, t, -1)
        video_body_features = self.temporal_net_body(image_body_features)

        # Concatenate the two parts
        video_features = torch.cat((video_face_features, video_body_features), dim=-1)
        video_features = self.project_fc(video_features)
        
        # Fuse Gaze Features
        if gaze_features is not None:
            gaze_avg = gaze_features.mean(dim=1)
            gaze_encoded = self.gaze_mlp(gaze_avg.type(self.dtype))
            video_features = video_features + self.alpha_gaze * gaze_encoded
        
        # Keep raw features for classifier head (before L2 norm kills gradient diversity)
        video_features_raw = video_features
            
        # Robust normalization to avoid NaN on MPS (for CLIP similarity path)
        video_features = video_features / (video_features.norm(dim=-1, keepdim=True) + 1e-6)

        ################# Text Part ###################
        # Learnable prompts
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        
        # FORCE FP32 for Text Encoder to avoid NaN on MPS
        with torch.amp.autocast('cuda', enabled=False):
            # Text Encoder might contain layers incompatible with AMP on MPS or just unstable
            text_features = self.text_encoder(prompts, tokenized_prompts)
            # Robust normalization
            text_features = text_features.float() # Ensure float32
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)

        # Hand-crafted prompts (for MI Loss, not used for classification)
        hand_crafted_prompts = self.hand_crafted_prompt_embeddings
        tokenized_hand_crafted_prompts = self.tokenized_hand_crafted_prompts.to(hand_crafted_prompts.device)
        
        with torch.amp.autocast('cuda', enabled=False):
            hand_crafted_text_features = self.text_encoder(hand_crafted_prompts, tokenized_hand_crafted_prompts)
            hand_crafted_text_features = hand_crafted_text_features.float()
            # Robust normalization
            hand_crafted_text_features = hand_crafted_text_features / (hand_crafted_text_features.norm(dim=-1, keepdim=True) + 1e-6)

        ################# MoCo Updates ###################
        returned_moco_features = None
        if self.training and hasattr(self.args, 'use_moco') and self.args.use_moco:
            with torch.no_grad():
                self._momentum_update_key_encoder()
                k_video_features = self.forward_momentum(image_face, image_body, gaze_features=gaze_features)

            self._dequeue_and_enqueue(k_video_features)
            # Return video_features so trainer can compute Supervised MoCoRank
            returned_moco_features = video_features

        ################# Classification ###################
        if self.use_classifier_head:
            # CosineClassifier: normalizes internally, tau scales output
            output = self.classifier_head(video_features_raw)
        elif self.is_ensemble:
            # Reshape text features for ensembling: (C*P, D) -> (C, P, D)
            text_features = text_features.view(self.num_classes, self.num_prompts_per_class, -1)
            # Normalize again just in case (optional but safe) - Robust version
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)
            
            # Compute logits per prompt: (B, D) @ (D, P, C) -> (B, P, C)
            # Note: We use einsum for clarity with batch and ensemble dimensions
            logits = torch.einsum('bd,cpd->bcp', video_features, text_features)
            
            # Average the logits across the prompts for each class
            output = torch.mean(logits, dim=2) / self.args.temperature
        else:
            output = video_features @ text_features.t() / self.args.temperature

        return output, text_features, hand_crafted_text_features, returned_moco_features