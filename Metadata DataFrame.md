# Creating DICOM DataFrames

It's really handy to have all the DICOM info available in a single DataFrame, so let's create that! In this notebook, we'll just create the DICOM DataFrames. To see how to use them to analyze the competition data, see the follow-up notebook.

First, we'll install the latest versions of PyTorch and fastai v2 (not officially released yet) so we can use the fastai medical imaging module.

```bash
!pip install torch torchvision feather-format kornia pyarrow --upgrade > /dev/null
!pip install git+https://github.com/fastai/fastai_dev > /dev/null
```
Note: If you encounter errors such as missing packages (e.g., flaky, responses), you may need to install them manually.

Import necessary libraries:

```bash
from fastai2.basics import *
from fastai2.medical.imaging import *

```
Define the path to the dataset:

```bash
path = Path('../input/rsna-intracranial-hemorrhage-detection/')
Let's take a look at what files we have in the dataset:
```

```bash
path_trn = path/'stage_1_train_images'
fns_trn = path_trn.ls()
fns_trn[:5].attrgot('name')  # Display names of the first 5 files
```
Similarly, list files in the test set:

```bash

path_tst = path/'stage_1_test_images'
fns_tst = path_tst.ls()
```
Print the number of files in each dataset:

```bash
print(len(fns_trn), len(fns_tst))
```
We can grab a file and take a look inside using the dcmread method that fastai v2 adds:

```bash
fn = fns_trn[0]
dcm = fn.dcmread()
print(dcm)
```
# Labels

Before we pull the metadata out of the DICOM files, let's process the labels into a convenient format and save it for later. We'll use feather format because it's lightning fast!

```bash
def save_lbls():
    path_lbls = path/'stage_1_train.csv'
    lbls = pd.read_csv(path_lbls)
    lbls[["ID", "htype"]] = lbls.ID.str.rsplit("_", n=1, expand=True)
    lbls.drop_duplicates(['ID', 'htype'], inplace=True)
    pvt = lbls.pivot('ID', 'htype', 'Label')
    pvt.reset_index(inplace=True)    
    pvt.to_feather('labels.fth')
```

```bash
save_lbls()
df_lbls = pd.read_feather('labels.fth').set_index('ID')
print(df_lbls.head(8))
print(df_lbls.mean())

```
Clean up memory as we go:

```bash
del(df_lbls)
import gc; gc.collect()

```
# DICOM Meta

To turn the DICOM file metadata into a DataFrame, we can use the from_dicoms function that fastai v2 adds. By passing px_summ=True, summary statistics of the image pixels (mean/min/max/std) will be added to the DataFrame as well (although it takes much longer if you include this, since the image data has to be uncompressed).

```bash
%time df_tst = pd.DataFrame.from_dicoms(fns_tst, px_summ=True)
df_tst.to_feather('df_tst.fth')
print(df_tst.head())
```
Note: The process may take a while. The dataset may include corrupted files; such files are identified but not included in the final DataFrame.

Clean up memory:

```bash
del(df_tst)
gc.collect()
```
Repeat the process for the training set:

```bash
%time df_trn = pd.DataFrame.from_dicoms(fns_trn, px_summ=True)
df_trn.to_feather('df_trn.fth')

```
