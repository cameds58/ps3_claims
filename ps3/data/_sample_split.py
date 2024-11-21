import hashlib
import numpy as np


def hash_id_to_group(id_value):
    # Convert ID to hash and use modulo to get the group
    hash_value = int(hashlib.md5(str(id_value).encode()).hexdigest(), 16)
    return hash_value % 100

# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation.
def create_sample_split(df, id_column, training_frac=0.8):
    """Create sample split based on ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_column : str
        Name of ID column
    training_frac : float, optional
        Fraction to use for training, by default 0.9

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split based on IDs.
    """
    if not 0 < training_frac < 1:
        raise ValueError("training_frac must be between 0 and 1.")
    df['sample'] = df[id_column].apply(hash_id_to_group)
    df['sample'] = np.where(df['sample'] < training_frac * 100, 'train', 'test')

    return df
