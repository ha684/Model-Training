mlflow server --host 0.0.0.0 --port 5000

ngrok http 5000 --host-header="localhost:5000" --url https://smart-kind-macaque.ngrok-free.app

ngrok http 5000 --domain=smart-kind-macaque.ngrok-free.app --host-header=localhost:5000
