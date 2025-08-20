from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata
import pandas as pd
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
from sdv.evaluation.single_table import get_column_plot

metadata =  Metadata.load_from_json(
    filepath='metadata.json')

#load real_data

real_data = pd.read_csv('your_real_train_data.csv')

synthesizer = CTGANSynthesizer(
    metadata,  # required
    enforce_rounding=False,
    epochs=200,
    verbose=True
)

# Step 2: Train the synthesizer
synthesizer.fit(real_data)

# Step 3: Generate synthetic data
synthetic_data = synthesizer.sample(num_rows=1000) #define as needed
#save the data if needed
synthetic_data.to_csv('syn_trip_ctgan.csv', index=False)

synthesizer.get_loss_values()
fig1 = synthesizer.get_loss_values_plot()
fig1.show()
#save the model if needed
synthesizer.save(
    filepath='ModelSyn_ctgan.pkl'
)

# 1. perform basic validity checks
diagnostic = run_diagnostic(real_data, synthetic_data, metadata)

# 2. measure the statistical similarity
quality_report = evaluate_quality(real_data, synthetic_data, metadata)

quality_report.get_details(property_name='Column Shapes')

quality_report.get_details(property_name='Column Shapes').to_excel('report_ctgan_1.xlsx')

quality_report.get_details(property_name='Column Pair Trends').to_excel('report_ctgan_2.xlsx')

# 3. plot the data
fig2 = get_column_plot(
    real_data=real_data,
    synthetic_data=synthetic_data,
    metadata=metadata,
    column_name='...'
)

fig2.show()