This submodule is for supporting the APPFLx - APPFL as a service.

```bash
docker build --platform=linux/amd64 --no-cache --progress=plain -f ./src/appfl/service/Dockerfile -t myimage .
```

```bash
docker run --openid_token ... --compute_token ...
```