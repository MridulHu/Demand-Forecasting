import numpy as np

def calculate_inventory_metrics(preds, lead_time=3, service_level=1.65):

    demand_std = np.std(preds[:lead_time])
    safety_stock = service_level * demand_std

    reorder_point = lead_time * np.mean(preds[:lead_time]) + safety_stock

    return safety_stock, reorder_point