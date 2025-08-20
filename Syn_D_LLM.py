from tabula import Tabula
import pandas as pd
import torch
import os
# please refer to the following citation for more details.
'''
@inproceedings{zhao2025tabula,
  title={Tabula: Harnessing language models for tabular data synthesis},
  author={Zhao, Zilong and Birke, Robert and Chen, Lydia Y},
  booktitle={Pacific-Asia Conference on Knowledge Discovery and Data Mining},
  pages={247--259},
  year={2025},
  organization={Springer}
}
'''

data = pd.read_csv("your_real_data.csv")


categorical_columns = ['..', '...', '...']
#choose your preferred model
model = Tabula(llm='distilgpt2', experiment_dir="trip_training", batch_size=32, epochs=400,
               categorical_columns=categorical_columns)

if not os.path.exists("trip_training/..."):
    model.fit(data)
    torch.save(model.model.state_dict(), "trip_training/...")
else:
    model.model.load_state_dict(torch.load("trip_training/...."))
    model.columns = list(data.columns)
    model.num_cols = [_ for _ in list(model.columns) if _ not in categorical_columns]
    model.encode_categorical_column(data)

synthetic_data = model.sample(n_samples=47283)
synthetic_data.to_csv("trip_training/...", index=False)