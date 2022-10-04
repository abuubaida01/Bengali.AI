import os
import ast
from model_dispatcher import MODEL_DISPATHER
from dataset import BengaliDatasetTrain
import torch
import torch.nn as nn
# import tez
from tqdm import tqdm


# Variables
DEVICE = 'cpu'
TRAINING_FOLDS_CSV= "../input/train_folds.csv"

IMG_HEIGHT = 137
IMG_WIDTH = 236
EPOCHS= 10
BASE_MODEL= "resnet34"

TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 8

MODEL_MEAN = (0.485, 0.456, 0.406)
MODEL_STD = (0.229, 0.224, 0.225)



# Loss Function 
def loss_fn(outputs, targets):
    o1, o2, o3 = outputs
    t1, t2, t3 = targets
    l1 = nn.CrossEntropyLoss()(o1, t1)
    l2 = nn.CrossEntropyLoss()(o2, t2)
    l3 = nn.CrossEntropyLoss()(o3, t3)
    return (l1 +  l2 + l3 )/ 3 # average loss 


# Training Function 
def train(dataset, data_loader, model, optimizer):
    model.train()
    for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset)/data_loader.batch_size)):
        image = d['image']
        grapheme = d['grapheme_root'] 
        vowel_diacritic = d['vowel_diacritic'] 
        consonant_diacritic = d['consonant_diacritic'] 

        image = image.to(DEVICE, dtype=torch.float)
        grapheme = grapheme.to(DEVICE, dtype=torch.long)
        vowel_diacritic = vowel_diacritic .to(DEVICE, dtype=torch.long)
        consonant_diacritic = consonant_diacritic.to(DEVICE, dtype=torch.long)

        optimizer.zero_grad()
        # need to be in same order graphem, vowel, consonants 
        outputs = model(image)
        targets = (grapheme, vowel_diacritic, consonant_diacritic)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

# Validation Function 
def evaluate(dataset, data_loader, model):
    model.eval()
    final_loss = 0
    counter = 0
    for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset)/data_loader.batch_size)):
        counter+=1
        image = d['image']
        grapheme = d['grapheme_root'] 
        vowel_diacritic = d['vowel_diacritic'] 
        consonant_diacritic = d['consonant_diacritic'] 

        image = image.to(DEVICE, dtype=torch.float)
        grapheme = grapheme.to(DEVICE, dtype=torch.long)
        vowel_diacritic = vowel_diacritic .to(DEVICE, dtype=torch.long)
        consonant_diacritic = consonant_diacritic.to(DEVICE, dtype=torch.long)

        # need to be in same order graphem, vowel, consonants 
        outputs = model(image)
        targets = (grapheme, vowel_diacritic, consonant_diacritic)
        loss = loss_fn(outputs, targets)
        final_loss += loss
    return final_loss/counter
lot = [(0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 4, 3),(0, 4, 2, 3),(4, 1, 2, 3)]
lov = [(4,), (3,), (2,), (1,), (0,)]

for i in range(5):
    TRAINING_FOLDS= lot[i]
    VALIDATION_FOLDS= lov[i]

def main():
    model = MODEL_DISPATHER[BASE_MODEL](pretrained=True)
    model.to(DEVICE)

    train_dataset = BengaliDatasetTrain(
        folds=TRAINING_FOLDS,
        img_height= IMG_HEIGHT,
        img_width= IMG_WIDTH,
        mean=MODEL_MEAN,
        std= MODEL_STD
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size = TRAIN_BATCH_SIZE,
        shuffle = True, 
        num_workers=4
    )

    valid_dataset = BengaliDatasetTrain(
        folds= VALIDATION_FOLDS,
        img_height= IMG_HEIGHT,
        img_width= IMG_WIDTH,
        mean= MODEL_MEAN,
        std= MODEL_STD
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size = TEST_BATCH_SIZE,
        shuffle = False, # doesn't matter, already Shuffled
        num_workers=4
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=5, factor=0.3, verbose=True)

    # for using multiple GPUs (which i don't have)
    # if torch.cuda.device_count >1:
    #     model = nn.DataParallel()
    
    for epoch in range(EPOCHS):
        train(train_dataset, train_loader, model, optimizer)
        val_score = evaluate(train_dataset, valid_loader, model)
        scheduler.step(val_score)
        torch.save(model.state_dict(), f'{BASE_MODEL}_fold {VALIDATION_FOLDS[0]}.bin')

if __name__ == '__main__':
    main()


