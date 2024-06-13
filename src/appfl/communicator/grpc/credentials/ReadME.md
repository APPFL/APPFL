# SSL/TLS Credentials
**All credential files are examples provided by [gRPC official documentations](https://github.com/grpc/grpc/tree/master/examples/python/auth/credentials), and they are only used for demonstration and testing purposes.**

Specifically, 
- `localhost.crt` is the public SSL certificate for encrypting data sent to the server (certificate holder).
- `localhost.key` is the **secret** key that works with the public key to encrypt and decrypt data. In practice, this file must be **kept secure** to maintain the integrity of the encrypted communication (however, for this usecase, it is fine to publicly release it for demo).
- `root.crt` is the root SSL certificate issued by a trusted certificate authority (CA) for signing other certificates. It is used by the client to validate the server SSL certificate. 