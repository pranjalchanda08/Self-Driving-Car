import utils.utils as ut

# Step 1: Import data
data = ut.import_data_info()

# Step 2: Get Balanced data
data = ut.get_balance_data(data, True)

