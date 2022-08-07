import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from model.model import ShelterOutcomeModel
from dataloader.dataloaders import Dataset, ShelterOutcomeDataset, DeviceDataloader

def main():
    print("Data loading...")
    train_data = pd.read_csv("data/train.csv")
    test_X = pd.read_csv("data/test.csv")
    print("Data loading - Done")

    print("Data processing...")
    X = train_data.drop(columns=['AnimalID', 'OutcomeType', 'OutcomeSubtype'])
    stacked_df = X.append(test_X.drop(columns=['ID']))
    stacked_df['DateTime'] = pd.to_datetime(stacked_df['DateTime'])
    stacked_df['year'] = stacked_df['DateTime'].dt.year
    stacked_df['month'] = stacked_df['DateTime'].dt.month
    stacked_df = stacked_df.drop(columns=['DateTime'])

    for col in stacked_df.columns:
        n_nulls = stacked_df[col].isnull().sum()
        if n_nulls > 10000:
            print(f"Drop column {col} with {n_nulls} nulls")
            stacked_df = stacked_df.drop(columns=[col])

    for col in stacked_df.columns:
        if stacked_df.dtypes[col] == 'object':
            stacked_df[col] = stacked_df[col].fillna('NA')
        else:
            stacked_df[col] = stacked_df[col].fillna(0)
        stacked_df[col] = LabelEncoder().fit_transform(stacked_df[col])

    for col in stacked_df.columns:
        stacked_df[col] = stacked_df[col].astype('category')

    test_X = stacked_df[train_data.shape[0]:]

    embedded_cols = dict()
    for name, col in stacked_df.items():
        n_categories = len(col.cat.categories)
        if n_categories > 2:
            embedded_cols[name] = n_categories
    embedded_cols_names = embedded_cols.keys()
    embedding_sizes = [(n_categories, min(50, (n_categories+1)//2)) for _, n_categories in embedded_cols.items()]
    print("Data processing - Done")
    print(test_X.columns)
    print("DataLoaders creating...")
    batch_size = 1000
    test_dataset = ShelterOutcomeDataset(test_X, np.zeros(len(test_X)), embedded_cols_names)
    test_dataloader = DataLoader(test_dataset, batch_size)
    print("DataLoaders creating - Done")

    print("Model loading...")
    n_cont = len(test_X.columns) - len(embedded_cols_names)
    model = ShelterOutcomeModel(embedding_sizes, n_cont)
    model = torch.load("saved/models/saved_model.pt")
    print("Model loading - Done")

    print("Model referencing...")
    preds = list()
    with torch.no_grad():
        for x_cat, x_cont, y in test_dataloader:
            output = model(x_cat, x_cont)
            prob = F.softmax(output, dim=1)
            preds.append(prob)

    final_probs = [item for sublist in preds for item in sublist]

    submission = pd.DataFrame(columns=['ID', 'Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer'])
    submission['ID'] = list(range(1, len(final_probs) + 1))
    submission['Adoption'] = [float(t[0]) for t in final_probs]
    submission['Died'] = [float(t[1]) for t in final_probs]
    submission['Euthanasia'] = [float(t[2]) for t in final_probs]
    submission['Return_to_owner'] = [float(t[3]) for t in final_probs]
    submission['Transfer'] = [float(t[4]) for t in final_probs]

    submission.to_csv('saved/submissions/submission.csv', index=False)
    print("Model referencing - Done")

if __name__ == '__main__':
    main()