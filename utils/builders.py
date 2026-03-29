# builders.py

import argparse
from typing import Tuple
import os
import torch
import torch.utils.data
from clip import clip

from dataloader.caer_s_dataloader import CAERSDataset, caers_train_data_loader, caers_val_data_loader, caers_test_data_loader
from models.Generate_Model import GenerateModel
from models.Text import *
from utils.utils import *


def build_model(args: argparse.Namespace, input_text: list) -> torch.nn.Module:
    print("Loading pretrained CLIP model...")
    CLIP_model, _ = clip.load(args.clip_path, device='cpu')

    print("\nInput Text Prompts:")
    # Handle the case where input_text is a list of lists for prompt ensembling
    if any(isinstance(i, list) for i in input_text):
        for class_prompts in input_text:
            print(f"- Class: {class_prompts}")
    else:
        for text in input_text:
            print(text)


    print("\nInstantiating GenerateModel...")
    model = GenerateModel(input_text=input_text, clip_model=CLIP_model, args=args)

    for name, param in model.named_parameters():
        param.requires_grad = False

    # Freeze CLIP image encoder if lr_image_encoder is 0
    # Otherwise, make it trainable.
    if args.lr_image_encoder > 0:
        for name, param in model.named_parameters():
            if "image_encoder" in name:
                param.requires_grad = True

    trainable_params_keywords = ["temporal_net", "prompt_learner", "temporal_net_body", "project_fc", "face_adapter"]
    
    print('\nTrainable parameters:')
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in trainable_params_keywords):
            param.requires_grad = True
            print(f"- {name}")
    print('************************\n')

    return model


def get_class_info(args: argparse.Namespace) -> Tuple[list, list]:
    """
    Returns class_names and input_text for the dataset.
    """
    dataset_name = args.dataset.strip()
    print(f"DEBUG: get_class_info processing dataset '{dataset_name}'")

    if dataset_name == "CAER" or dataset_name == "CAER-S":
        class_names = class_names_caer
        class_names_with_context = class_names_with_context_caer
        class_descriptor = class_descriptor_caer
        ensemble_prompts = prompt_ensemble_caer
    elif dataset_name == "DAiSEE":
        class_names = class_names_daisee
        class_names_with_context = class_names_with_context_daisee
        class_descriptor = class_descriptor_daisee
        ensemble_prompts = prompt_ensemble_daisee
    elif dataset_name == "DAiSEE4Level":
        class_names = class_names_daisee_4level
        class_names_with_context = class_names_with_context_daisee_4level
        class_descriptor = class_descriptor_daisee_4level
        ensemble_prompts = prompt_ensemble_daisee_4level
    elif dataset_name == "DAiSEE4Discrete":
        class_names = class_names_daisee4
        class_names_with_context = class_names_with_context_daisee4
        class_descriptor = class_descriptor_daisee4
        ensemble_prompts = prompt_ensemble_daisee4
    elif dataset_name == "StudentEngagement":
        class_names = class_names_student_engagement
        class_names_with_context = class_names_with_context_student_engagement
        class_descriptor = class_descriptor_student_engagement
        ensemble_prompts = prompt_ensemble_student_engagement
    elif dataset_name == "StudentEngagement6":
        class_names = class_names_student_engagement_6
        class_names_with_context = class_names_with_context_student_engagement_6
        class_descriptor = class_descriptor_student_engagement_6
        ensemble_prompts = prompt_ensemble_student_engagement_6
    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' is not implemented. Only CAER-S and DAiSEE are supported in this version.")

    if args.text_type == "class_names":
        input_text = class_names
    elif args.text_type == "class_names_with_context":
        input_text = class_names_with_context
    elif args.text_type == "class_descriptor":
        input_text = class_descriptor
    elif args.text_type == "prompt_ensemble":
        input_text = ensemble_prompts
    else:
        raise ValueError(f"Unknown text_type: {args.text_type}")

    return class_names, input_text



def build_dataloaders(args: argparse.Namespace) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]: 
    train_annotation_file_path = args.train_annotation
    val_annotation_file_path = args.val_annotation
    test_annotation_file_path = args.test_annotation
    
    class_names, _ = get_class_info(args)
    num_classes = len(class_names)

    # Debug print
    print(f"DEBUG: args.dataset = '{args.dataset}'")

    if args.dataset.strip() == "CAER-S":
        print(f"=> Using CAER-S specific dataloader...")
        
        # Pass args.bounding_box_face as the JSON file for faces
        bbox_json = args.bounding_box_face
        print(f"=> Face Bounding Box JSON: {bbox_json}")

        # Instantiate dataset to support weighted sampler
        train_data = CAERSDataset(
            root_dir=args.root_dir, 
            list_file=train_annotation_file_path, 
            mode='train', 
            image_size=args.image_size, 
            bounding_box_json=bbox_json
        )
        
        val_data = CAERSDataset(
            root_dir=args.root_dir, 
            list_file=val_annotation_file_path, 
            mode='val', 
            image_size=args.image_size, 
            bounding_box_json=bbox_json
        )
        
        test_data = CAERSDataset(
            root_dir=args.root_dir, 
            list_file=test_annotation_file_path, 
            mode='test', 
            image_size=args.image_size, 
            bounding_box_json=bbox_json
        )
        
        sampler = None
        shuffle = True
        if args.use_weighted_sampler:
            print("=> Using WeightedRandomSampler for CAER-S.")
            targets = [s[1] for s in train_data.samples]
            class_counts = torch.tensor([targets.count(i) for i in range(num_classes)])
            class_counts = torch.where(class_counts == 0, torch.ones_like(class_counts), class_counts)
            class_weights = 1. / class_counts.float()
            sample_weights = [class_weights[t] for t in targets]
            sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
            shuffle = False
            
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
            num_workers=args.workers, pin_memory=True, drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )

        print(f"Total number of training images: {len(train_data)}")
        return train_loader, val_loader, test_loader

    elif args.dataset.strip() == "DAiSEE":
        print(f"=> Using DAiSEE specific dataloader...")
        from dataloader.daisee_dataloader import DAiSEEDataset
        
        max_samples = getattr(args, 'max_samples_per_class', 0)
        train_data = DAiSEEDataset(
            root_dir=args.root_dir,
            annotation_file=train_annotation_file_path,
            mode='train',
            num_segments=args.num_segments,
            duration=args.duration,
            image_size=args.image_size,
            max_samples_per_class=max_samples,
            merge_3class=True
        )
        
        val_data = DAiSEEDataset(
            root_dir=args.root_dir,
            annotation_file=val_annotation_file_path,
            mode='val',
            num_segments=args.num_segments,
            duration=args.duration,
            image_size=args.image_size,
            merge_3class=True
        )
        
        test_data = DAiSEEDataset(
            root_dir=args.root_dir,
            annotation_file=test_annotation_file_path,
            mode='test',
            num_segments=args.num_segments,
            duration=args.duration,
            image_size=args.image_size,
            merge_3class=True
        )
        
        sampler = None
        shuffle = True
        if args.use_weighted_sampler:
            print("=> Using WeightedRandomSampler for DAiSEE.")
            targets = [s[1] for s in train_data.samples]
            class_counts = torch.tensor([targets.count(i) for i in range(num_classes)])
            class_counts = torch.where(class_counts == 0, torch.ones_like(class_counts), class_counts)
            class_weights = 1. / class_counts.float()
            sample_weights = [class_weights[t] for t in targets]
            sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
            shuffle = False

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
            num_workers=args.workers, pin_memory=True, drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
        print(f"Total number of training samples: {len(train_data)}")
        return train_loader, val_loader, test_loader

    elif args.dataset.strip() == "DAiSEE4Level":
        print(f"=> Using DAiSEE 4-Level Engagement dataloader...")
        from dataloader.daisee_dataloader import DAiSEEDataset
        
        max_samples = getattr(args, 'max_samples_per_class', 0)
        train_data = DAiSEEDataset(
            root_dir=args.root_dir,
            annotation_file=train_annotation_file_path,
            mode='train',
            num_segments=args.num_segments,
            duration=args.duration,
            image_size=args.image_size,
            max_samples_per_class=max_samples,
            merge_3class=False
        )
        val_data = DAiSEEDataset(
            root_dir=args.root_dir,
            annotation_file=val_annotation_file_path,
            mode='val',
            num_segments=args.num_segments,
            duration=args.duration,
            image_size=args.image_size,
            merge_3class=False
        )
        test_data = DAiSEEDataset(
            root_dir=args.root_dir,
            annotation_file=test_annotation_file_path,
            mode='test',
            num_segments=args.num_segments,
            duration=args.duration,
            image_size=args.image_size,
            merge_3class=False
        )
        
        sampler = None
        shuffle = True
        if args.use_weighted_sampler:
            print("=> Using WeightedRandomSampler for DAiSEE4Level.")
            targets = [s[1] for s in train_data.samples]
            class_counts = torch.tensor([targets.count(i) for i in range(num_classes)])
            class_counts = torch.where(class_counts == 0, torch.ones_like(class_counts), class_counts)
            class_weights = 1. / class_counts.float()
            sample_weights = [class_weights[t] for t in targets]
            sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
            shuffle = False

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
            num_workers=args.workers, pin_memory=True, drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
        print(f"Total number of training samples: {len(train_data)}")
        return train_loader, val_loader, test_loader
        
    elif args.dataset.strip() == "DAiSEE4Discrete":
        print(f"=> Using DAiSEE 4-Discrete dataloader...")
        from dataloader.daisee_dataloader import DAiSEE4DiscreteDataset
        
        extra_files = getattr(args, 'extra_train_annotations', [])
        train_data = DAiSEE4DiscreteDataset(
            root_dir=args.root_dir,
            annotation_file=train_annotation_file_path,
            mode='train',
            num_segments=args.num_segments,
            duration=args.duration,
            image_size=args.image_size,
            extra_annotation_files=extra_files
        )
        val_data = DAiSEE4DiscreteDataset(
            root_dir=args.root_dir,
            annotation_file=val_annotation_file_path,
            mode='val',
            num_segments=args.num_segments,
            duration=args.duration,
            image_size=args.image_size
        )
        test_data = DAiSEE4DiscreteDataset(
            root_dir=args.root_dir,
            annotation_file=test_annotation_file_path,
            mode='test',
            num_segments=args.num_segments,
            duration=args.duration,
            image_size=args.image_size
        )
        
        sampler = None
        shuffle = True
        if args.use_weighted_sampler:
            print("=> Using WeightedRandomSampler for DAiSEE4Discrete.")
            targets = [s[1] for s in train_data.samples]
            class_counts = torch.tensor([targets.count(i) for i in range(num_classes)])
            class_counts = torch.where(class_counts == 0, torch.ones_like(class_counts), class_counts)
            class_weights = 1. / class_counts.float()
            sample_weights = [class_weights[t] for t in targets]
            sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
            shuffle = False

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
            num_workers=args.workers, pin_memory=True, drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
        print(f"Total number of training samples: {len(train_data)}")
        return train_loader, val_loader, test_loader

    elif args.dataset.strip() == "StudentEngagement":
        print(f"=> Using StudentEngagement specific dataloader...")
        from dataloader.student_engagement_dataloader import StudentEngagementDataset
        
        train_data = StudentEngagementDataset(
            root_dir=args.root_dir,
            mode='train',
            num_segments=args.num_segments,
            image_size=args.image_size
        )
        
        val_data = StudentEngagementDataset(
            root_dir=args.root_dir,
            mode='val',
            num_segments=args.num_segments,
            image_size=args.image_size
        )
        
        test_data = StudentEngagementDataset(
            root_dir=args.root_dir,
            mode='test',
            num_segments=args.num_segments,
            image_size=args.image_size
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
        print(f"Total number of training samples: {len(train_data)}")
        return train_loader, val_loader, test_loader

    elif args.dataset.strip() == "StudentEngagement6":
        print(f"=> Using StudentEngagement 6-class dataloader...")
        from dataloader.student_engagement_dataloader import StudentEngagement6Dataset
        
        train_data = StudentEngagement6Dataset(
            root_dir=args.root_dir,
            mode='train',
            num_segments=args.num_segments,
            image_size=args.image_size
        )
        val_data = StudentEngagement6Dataset(
            root_dir=args.root_dir,
            mode='val',
            num_segments=args.num_segments,
            image_size=args.image_size
        )
        test_data = StudentEngagement6Dataset(
            root_dir=args.root_dir,
            mode='test',
            num_segments=args.num_segments,
            image_size=args.image_size
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
        print(f"Total number of training samples: {len(train_data)}")
        return train_loader, val_loader, test_loader

    else:
        raise NotImplementedError(f"Dataset {args.dataset} is not supported.")
