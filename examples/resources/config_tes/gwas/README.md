docker build --no-cache -f examples/resources/config_tes/gwas/Dockerfile -t appfl/client-gwas:latest .

docker buildx build --platform linux/amd64  -f examples/resources/config_tes/gwas/Dockerfile -t zilinghan/client-gwas:latest  --push .      

python tes/run.py --server_config ./resources/config_tes/gwas/server.yaml --client_config ./resources/config_tes/gwas/clients.yaml