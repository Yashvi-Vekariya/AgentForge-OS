import logging
import mlflow
import mlflow.transformers
from typing import Dict, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class MLflowManager:
    """MLflow manager for experiment tracking and model registry"""
    
    def __init__(self, tracking_uri: str = "mlruns", experiment_name: str = "multi-agent-system"):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow tracking"""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"MLflow setup complete: {self.tracking_uri}, experiment: {self.experiment_name}")
        except Exception as e:
            logger.error(f"Error setting up MLflow: {str(e)}")
    
    def start_run(self, run_name: str = None, tags: Dict[str, Any] = None):
        """Start a new MLflow run"""
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        mlflow.start_run(run_name=run_name, tags=tags)
        logger.info(f"Started MLflow run: {run_name}")
    
    def end_run(self):
        """End current MLflow run"""
        mlflow.end_run()
        logger.info("Ended MLflow run")
    
    def log_agent_metrics(self, 
                         agent_name: str,
                         task: str,
                         metrics: Dict[str, float],
                         parameters: Dict[str, Any] = None):
        """Log agent execution metrics"""
        try:
            # Add agent-specific tags
            tags = {
                "agent_name": agent_name,
                "task_type": task,
                "timestamp": datetime.now().isoformat()
            }
            
            self.start_run(f"agent_{agent_name}", tags)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log parameters
            if parameters:
                for param_name, param_value in parameters.items():
                    if isinstance(param_value, (int, float, str, bool)):
                        mlflow.log_param(param_name, param_value)
                    else:
                        mlflow.log_param(param_name, str(param_value))
            
            # Log task description
            mlflow.log_text(task, "task_description.txt")
            
            self.end_run()
            logger.info(f"Logged metrics for {agent_name}")
            
        except Exception as e:
            logger.error(f"Error logging agent metrics: {str(e)}")
            self.end_run()
    
    def log_model_performance(self,
                            model_name: str,
                            model_type: str,
                            performance_metrics: Dict[str, float],
                            dataset_info: Dict[str, Any] = None):
        """Log model performance metrics"""
        try:
            tags = {
                "model_name": model_name,
                "model_type": model_type,
                "evaluation_timestamp": datetime.now().isoformat()
            }
            
            if dataset_info:
                tags.update(dataset_info)
            
            self.start_run(f"model_eval_{model_name}", tags)
            
            for metric_name, metric_value in performance_metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            self.end_run()
            logger.info(f"Logged performance metrics for {model_name}")
            
        except Exception as e:
            logger.error(f"Error logging model performance: {str(e)}")
            self.end_run()
    
    def log_training_run(self,
                        model_name: str,
                        training_config: Dict[str, Any],
                        training_metrics: Dict[str, float],
                        model_artifact_path: str = None):
        """Log training run details and artifacts"""
        try:
            tags = {
                "model_name": model_name,
                "training_type": "fine_tuning",
                "timestamp": datetime.now().isoformat()
            }
            
            self.start_run(f"training_{model_name}", tags)
            
            # Log training configuration
            for config_key, config_value in training_config.items():
                if isinstance(config_value, (int, float, str, bool)):
                    mlflow.log_param(config_key, config_value)
                else:
                    mlflow.log_param(config_key, str(config_value))
            
            # Log training metrics
            for metric_name, metric_value in training_metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model artifact if provided
            if model_artifact_path:
                mlflow.log_artifact(model_artifact_path)
            
            # Save training config as artifact
            config_artifact = {
                "training_config": training_config,
                "training_metrics": training_metrics,
                "model_name": model_name
            }
            
            with open("training_config.json", "w") as f:
                json.dump(config_artifact, f, indent=2)
            
            mlflow.log_artifact("training_config.json")
            
            self.end_run()
            logger.info(f"Logged training run for {model_name}")
            
        except Exception as e:
            logger.error(f"Error logging training run: {str(e)}")
            self.end_run()
    
    def register_model(self, 
                      model_name: str, 
                      model_path: str,
                      metadata: Dict[str, Any] = None):
        """Register model in MLflow model registry"""
        try:
            # Log model
            mlflow.transformers.log_model(
                transformer_model=model_path,
                artifact_path=model_name,
                registered_model_name=model_name
            )
            
            # Add metadata if provided
            if metadata:
                client = mlflow.tracking.MlflowClient()
                latest_version = client.get_latest_versions(model_name, stages=["None"])[0]
                client.update_model_version(
                    name=model_name,
                    version=latest_version.version,
                    description=json.dumps(metadata)
                )
            
            logger.info(f"Registered model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error registering model: {str(e)}")
    
    def get_experiment_results(self, experiment_name: str = None) -> Dict[str, Any]:
        """Get results from an experiment"""
        try:
            if experiment_name:
                mlflow.set_experiment(experiment_name)
            
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name(experiment_name or self.experiment_name)
            
            if not experiment:
                return {"error": "Experiment not found"}
            
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["attributes.start_time DESC"]
            )
            
            results = {
                "experiment_name": experiment_name or self.experiment_name,
                "total_runs": len(runs),
                "runs": []
            }
            
            for run in runs[:10]:  # Last 10 runs
                run_info = {
                    "run_id": run.info.run_id,
                    "run_name": run.data.tags.get('mlflow.runName', ''),
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "tags": run.data.tags
                }
                results["runs"].append(run_info)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting experiment results: {str(e)}")
            return {"error": str(e)}