
import io
import matplotlib.pyplot as plt

from comet_ml import Experiment
from typing import Dict
from PIL import Image

class CometMLLogger:
    def __init__(self):
        self.experiment = Experiment(api_key="iU4f44llKnowZwmrEo9wfR2ch",
                                     project_name="general", 
                                     workspace="yahyaalaamassoud",
                                     log_code=False,
                                     log_graph=False)
            
    def log_params(self, params: Dict[str, int]):
        self.experiment.log_parameters(params)
        
    def log_metric(self, metric_name, metric_val, step=None):
        self.experiment.log_metric(metric_name, metric_val, step=step)
        
    def log_metrics(self, metrics: Dict[str, float], step=None):
        self.experiment.log_metrics(metrics, step=step)
        
    def log_figure(self, figure_name: str, step: str):
        self.experiment.log_image(image_data=self.__savefig(), 
                                  name=figure_name, 
                                  step=step, 
                                  overwrite=False)
        
    def __savefig(self):
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return Image.open(io.BytesIO(buf.getvalue()))
        
        