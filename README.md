# Welcome to the classification project repo!

I want to thank you for providing a great opportunity to work on the classification project!

Please find the project description overview below.

## Project Development Overview

- The project was organized under **trunk-based development** practice where small and frequent updates are merged into a core trunk (main branch). The current practice is common for Continuous Integration and Development (CI/CD) of the Machine Learning (ML) project lifecycle.
  
- The project leverages the potential of DVC framework(https://dvc.org) to track the data versioning and pipeline stages in tandem with MLFlow framework(https://mlflow.org) to track ML experiments. The tandem of frameworks makes feasible to manage ML project lifecycle.
  
- The structure of the ML backtesting pipeline. Command in terminal: dvc dag
![DVC stages](/dvc-pipeline-stages.png "DVC pipeline stages")

- The experiment containing evaluation stage in MLFlow UI
![MLFlow experiment](/mlflow-experiments-tracking.png "MLFlow experiment for evaluation stage")

## Project Structure

- `./.dvc/`: Directory contains the local configuration file, cache location and other utilities that DVC needs to operate.
- `./config/`: Directory containing the yaml config files for managing backtesting pipeline stages.
- `./data/`: Directory containing the data managed by DVC pipeline stages.
- `./results/`: Directory to store the performance results and models.
- `./src/`: Directory containing the source code files to run the backtesting pipeline.
- `./dvc.yaml`: Config file contains the backbone of backtesting pipeline stages.
- `./jump_start_your_journey.ipynb`: The jupyter notebook contains the journey of exploring data and showcasing valuable insights.
- `./paper_dr_knn.ipynb`: The jupyter notebook contains the answers related to paper - Distributionally Robust Weighted k-Nearest Neighbors. I didn't include simulation results since I have some doubts about their validity and objectivity.
- `./requirements.txt`: python package requirements for the project.
- `./README-Technical Case Study.md`: solution documentation and instructions to obtain the results.
- `./README.md`: classification project solution overview.

## Preliminary steps to run the project

- Install requirements.txt
- Create a remote store for DVC (in our case it's local). Terminal cmd: `mkdir /tmp/dvc_local_store`
- Put classification_dataset into data folder. Since the DVC store is local, we need to put the input data manually, otherwise we could run terminal cmd: `dvc pull` command to pull all the required data for the project from remote store.
- Run terminal cmd: `dvc dag` to check pipeline stages for the project.
- Run terminal cmd: `dvc repro` to run pipeline backtesting stages and reproduce the results. When the process is completed, commit the changes and push the data by running the terminal cmd: `dvc push`
- Run terminal cmd: `mlflow ui` to run the MLFlow locally to track training and evaluation stages.
- For the deployment part: 
  - get into the `deployment` folder and run terminal cmd: `uvicorn main:app`
  - open the link (http://127.0.0.1:8000/docs#/default/inference_inference_post) to test inference request of deployed model.
  - Inference request example:
    - ![Inference-request](/inference-request.png "Inference request")
  - Inference response:
    - ![Inference-response](/inference-response.png "Inference response")