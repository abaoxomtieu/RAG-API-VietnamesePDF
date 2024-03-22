import os
import shutil

def remove_all_vectorstores(folder_path = "./vectorstores"):
    """
    Remove all files and subdirectories within the specified folder.

    Args:
    - folder_path (str): Path to the folder whose contents will be removed.
    """
    try:
        # Iterate over all files and subdirectories within the folder
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            # If the item is a file, remove it
            if os.path.isfile(item_path):
                os.remove(item_path)
            # If the item is a directory, remove it recursively
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        print(f"All contents within {folder_path} have been removed successfully.")
    except Exception as e:
        print(f"Error occurred while removing contents: {e}")
        
# remove_all_vectorstores()