import pandas as pd


def get_whole_dataset(path: str) -> pd.Dataframe:
    train_data_path = f"{path}train_essays.csv"
    another_train_prompt = f"{path}train_v2_drcat_02.csv"
    second_dataset = pd.read_csv(another_train_prompt)
    train_data = pd.read_csv(train_data_path)
    full_training_data = train_data.rename(columns={"generated": "label"})
    full_training_data = full_training_data[['text', 'label']]
    second_dataset = second_dataset[['text', 'label']]
    full_training_data = pd.concat([full_training_data, second_dataset], axis=0)
    return full_training_data