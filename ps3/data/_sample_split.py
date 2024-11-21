import hashlib
import numpy as np

def create_sample_split(df, id_column, training_frac=0.8):
    """
    Create a sample split based on ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_column : str
        Name of ID column
    training_frac : float, optional
        Fraction to use for training, by default 0.8

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split based on IDs.
    """
    if not 0 < training_frac < 1:
        raise ValueError("training_frac must be between 0 and 1.")

    # Generate stable hash for ID column
    def hash_id(id_value):
        if isinstance(id_value, str):
            hashed_value = int(hashlib.sha256(id_value.encode()).hexdigest(), 16)
        else:
            hashed_value = id_value
        return hashed_value

    # Apply hash function to the ID column
    df['hash'] = df[id_column].apply(hash_id)

    # Assign to train/test based on hash
    train_threshold = training_frac * 100
    df['sample'] = df['hash'] % 100
    df['sample'] = df['sample'].apply(lambda x: 'train' if x < train_threshold else 'test')

    # Drop intermediate hash column
    df.drop(columns=['hash'], inplace=True)

    return df
