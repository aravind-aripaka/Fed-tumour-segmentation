# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# --- U-Net Building Blocks (DoubleConv, Down, Up, OutConv - keep as they are) ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1); diffY = x2.size()[2] - x1.size()[2]; diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x): return self.conv(x)
# --- End of U-Net Building Blocks ---

class UNetSegmenterLitModel(pl.LightningModule):
    def __init__(self, in_channels=4, num_classes=3, learning_rate=1e-4, bilinear_upsampling=True):
        super().__init__()
        self.save_hyperparameters()

        # Ensure num_classes is an int, as it's used for iterating and model output
        try:
            self.current_num_classes = int(self.hparams.num_classes)
        except (TypeError, ValueError) as e:
            raise TypeError(f"num_classes must be convertible to an int, got {self.hparams.num_classes}. Error: {e}")

        self.inc = DoubleConv(self.hparams.in_channels, 64)
        self.down1 = Down(64, 128); self.down2 = Down(128, 256); self.down3 = Down(256, 512)
        factor = 2 if self.hparams.bilinear_upsampling else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, self.hparams.bilinear_upsampling)
        self.up2 = Up(512, 256 // factor, self.hparams.bilinear_upsampling)
        self.up3 = Up(256, 128 // factor, self.hparams.bilinear_upsampling)
        self.up4 = Up(128, 64, self.hparams.bilinear_upsampling)
        self.outc = OutConv(64, self.current_num_classes) # Use the int version

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        x_up = self.up1(x5, x4); x_up = self.up2(x_up, x3); x_up = self.up3(x_up, x2); x_up = self.up4(x_up, x1)
        logits = self.outc(x_up)
        return logits

    def _calculate_dice_score(self, preds_probs, targets_binary, smooth=1e-6):
        """Calculates Dice score for multi-label case, expects (B, C, H, W)"""
        # Flatten H, W dimensions
        preds_flat = preds_probs.view(preds_probs.shape[0], preds_probs.shape[1], -1) # (B, C, H*W)
        targets_flat = targets_binary.view(targets_binary.shape[0], targets_binary.shape[1], -1) # (B, C, H*W)

        intersection = (preds_flat * targets_flat).sum(dim=2) # Sum over H*W, result (B, C)
        sum_preds = preds_flat.sum(dim=2)  # (B, C)
        sum_targets = targets_flat.sum(dim=2) # (B, C)

        dice_per_batch_per_class = (2. * intersection + smooth) / (sum_preds + sum_targets + smooth) # (B, C)
        
        # Average Dice score per class over the batch
        dice_per_class = dice_per_batch_per_class.mean(dim=0) # (C,)
        
        # Macro average: average of Dice scores for each class
        # Filter out NaNs that can occur if a class has no positive instances in both preds and targets for entire batch
        valid_dice_scores = dice_per_class[~torch.isnan(dice_per_class)]
        if valid_dice_scores.numel() > 0:
            macro_dice = torch.mean(valid_dice_scores)
        else:
            macro_dice = torch.tensor(0.0, device=preds_probs.device) # Or torch.nan if preferred for "no valid classes"
            
        return macro_dice


    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.loss_fn(logits, masks.float())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.loss_fn(logits, masks.float()) # Ensure masks is float for BCE
        
        preds_probs = torch.sigmoid(logits) # Probabilities (B, C, H, W)
        
        # For Dice, we need binary predictions and binary targets
        preds_binary = (preds_probs > 0.5).float() # Apply threshold
        targets_binary = masks.float() # Ensure masks are float (should be 0.0 or 1.0)

        # --- Manual Dice Calculation ---
        dice = self._calculate_dice_score(preds_binary, targets_binary)
        # --- End Manual Dice ---

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_dice', dice, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer