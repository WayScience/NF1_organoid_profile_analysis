#!/usr/bin/env python
# coding: utf-8

# This notebook unzips the data compressed file and sets up the data directory for analysis.
# This way all users have the same data directory structure and files for analysis.

# In[1]:


import argparse
import pathlib
import shutil

from notebook_init_utils import bandicoot_check, init_notebook

root_dir, in_notebook = init_notebook()
bandicoot_path = pathlib.Path("~/mnt/bandicoot").expanduser().resolve()
profile_base_dir = bandicoot_check(bandicoot_path, root_dir)


# In[2]:


if not in_notebook:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    args = parser.parse_args()
    OVERWRITE = args.overwrite
else:
    OVERWRITE = True


# In[3]:


shippable_zip_path = pathlib.Path(
    f"{profile_base_dir}/data/shippable_dir.zip"
).resolve()
if not shippable_zip_path.exists():
    print("Shippable zip not found")
    raise FileNotFoundError(
        "Shippable zip not found, please download and place in the data directory (relative to the root of the git repository)."
    )


# In[4]:


unpack_dir = pathlib.Path(f"{root_dir}/data/").resolve(strict=False)
# check if unpack_dir is empty or not
if not OVERWRITE:
    print("Data directory is not empty and OVERWRITE is False. Skipping unpacking.")
else:
    # unzip the shippable_dir.zip file to the data directory
    shutil.unpack_archive(shippable_zip_path, unpack_dir, "zip")
