"""
DIA'GNOS HAND — Model definition
EfficientNetV2-S with custom classification head.
Architecture must match the training notebook exactly.
"""
import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

from app.config import NUM_CLASSES, MODEL_PATH


class BiomedCLIPClassifier(nn.Module):
    """
    Backbone BiomedCLIP (vision + texte) → projection → tête MLP de classification.

    Architecture :
        image  → vision_encoder  → img_emb (512d)
        texte  → text_encoder    → txt_emb (512d)
        concat → [img_emb ; txt_emb] (1024d) → MLP → logits
    """

    def __init__(self, clip_model, num_classes, embed_dim=512, dropout=0.3):
        super().__init__()
        self.clip_model = clip_model
        self.embed_dim = embed_dim

        # Tête MLP
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def encode_image(self, images):
        """Encode les images via le vision encoder CLIP."""
        return self.clip_model.encode_image(images)

    def encode_text(self, tokens):
        """Encode les tokens via le text encoder CLIP."""
        return self.clip_model.encode_text(tokens)

    def forward(self, images, tokens):
        img_emb = F.normalize(self.encode_image(images), dim=-1)
        txt_emb = F.normalize(self.encode_text(tokens), dim=-1)

        # Fusion par concaténation
        fused = torch.cat([img_emb, txt_emb], dim=-1)  # (B, 1024)
        logits = self.classifier(fused)
        return logits

    # ── Gestion du dégel ──
    def freeze_encoders(self):
        """Phase 1 : geler tout sauf la tête MLP."""
        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze_top_layers(self, n_layers=3):
        """Phase 2 : dégeler les n dernières couches transformer de chaque encodeur."""

        # ═══ Vision encoder (ViT) ═══
        visual = self.clip_model.visual
        if hasattr(visual, 'transformer'):
            blocks = visual.transformer.resblocks
        elif hasattr(visual, 'trunk'):
            blocks = visual.trunk.blocks
        else:
            blocks = [m for m in visual.modules()
                    if m.__class__.__name__ in ('ResidualAttentionBlock', 'Block')]

        if blocks:
            total_v = len(blocks)
            for block in blocks[-n_layers:]:
                for param in block.parameters():
                    param.requires_grad = True
            print(f"   Vision : dégel des {n_layers}/{total_v} derniers blocs")
        else:
            print("   ⚠️  Vision : blocs non détectés → dégel complet")
            for p in visual.parameters():
                p.requires_grad = True

        # ═══ Text encoder (PubMedBERT) ═══
        text_enc = self.clip_model.text
        bert_layers = None

        # PubMedBERT : text.transformer.encoder.layer
        if hasattr(text_enc, 'transformer'):
            transformer = text_enc.transformer
            if hasattr(transformer, 'encoder') and hasattr(transformer.encoder, 'layer'):
                bert_layers = transformer.encoder.layer
            elif hasattr(transformer, 'resblocks'):
                bert_layers = transformer.resblocks

        # Fallback : chercher tout module nommé "layer" qui est une ModuleList
        if bert_layers is None:
            for name, module in text_enc.named_modules():
                if isinstance(module, nn.ModuleList) and 'layer' in name:
                    bert_layers = module
                    break

        if bert_layers is not None:
            total_t = len(bert_layers)
            for layer in bert_layers[-n_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"   Texte  : dégel des {n_layers}/{total_t} dernières couches BERT")
        else:
            print("   ⚠️  Texte : couches non détectées → dégel complet")
            for p in text_enc.parameters():
                p.requires_grad = True

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

# def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
#     """
#     Reconstruct the exact same architecture used during training:
#       backbone  → EfficientNetV2-S (pretrained ImageNet)
#       classifier → Dropout(0.3) → Linear(1280, 256) → ReLU
#                    → Dropout(0.2) → Linear(256, num_classes)
#     """
#     model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)

#     in_features = model.classifier[1].in_features  # 1280
#     model.classifier = nn.Sequential(
#         nn.Dropout(p=0.3),
#         nn.Linear(in_features, 256),
#         nn.ReLU(),
#         nn.Dropout(p=0.2),
#         nn.Linear(256, num_classes),
#     )
#     return model

def build_model(device: torch.device, num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    EfficientNetV2-S backbone with custom classification head.
    Architecture must match training notebook exactly.
    """
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    return model.to(device), None  # no tokenizer needed

def load_model(device: torch.device) -> nn.Module:
    """Load trained weights and set the model to eval mode."""
    model, tokenizer = build_model(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    state = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer
