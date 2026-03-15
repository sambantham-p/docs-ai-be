# SecureDoc AI — Backend

> NestJS + TypeScript · PostgreSQL + pgvector · Gemini 1.5 · Double E2E Encryption

## What This Is

The backend for **SecureDoc AI** — a NestJS API server that powers a RAG (Retrieval-Augmented Generation) document assistant. All traffic is application-layer encrypted (RSA-OAEP + AES-GCM) on top of TLS. The server processes ciphertext, decrypts only what is needed in memory during Gemini prompt construction, and discards plaintext immediately after.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | NestJS + TypeScript |
| Database | PostgreSQL + pgvector extension |
| ORM | TypeORM (with migrations) |
| LLM | Gemini 1.5 Flash / Pro |
| Embedding | Gemini text-embedding-004 (768-dim vectors) |
| Encryption | Node.js `crypto.subtle` (W3C Web Crypto — no extra package) |
| Auth | JWT (access token 15m + refresh token 7d) + bcrypt (12 rounds) |
| PDF Parsing | pdf-parse + mammoth (for DOCX) |

---

## Module Architecture

```
AppModule
├── AuthModule         → Register, Login, JWT guards, blacklist
├── EncryptionModule   → Key exchange, EncryptionInterceptor (global)
├── DocumentsModule    → Upload, chunk, embed, sensitivity tagging
├── ChatModule         → Conversations, messages, RAG pipeline
└── AuditModule        → Decrypt event logging
```

### Request Flow

```
HTTP Request (encrypted)
    │
    ▼
EncryptionInterceptor (global)
  - validates clientId
  - RSA-OAEP decrypts session key
  - AES-GCM decrypts body → plain JSON
    │
    ▼
Controller  →  Service  →  DB / Gemini API
    │
    ▼
EncryptionInterceptor (response wrap)
  - RSA-OAEP wraps new session key with CLIENT public key
  - AES-GCM encrypts response
  - Returns { ed, esk, eiv }
```

---

## Database

**PostgreSQL** with the **pgvector** extension. All tables managed by TypeORM migrations.

| Table | Purpose |
|---|---|
| `users` | Auth identities |
| `encryption` | RSA key pairs (server) + registered client public keys |
| `documents` | Uploaded document metadata |
| `document_chunks` | Encrypted chunk text + 768-dim embedding vectors + sensitivity |
| `conversations` | Chat sessions linking user + documents |
| `messages` | Encrypted chat messages (user + AI) |
| `audit_logs` | Every chunk decrypt event with query hash |
| `jwt_blacklist` | Invalidated tokens on logout |

---

## Environment Variables

Create a `.env` file in the `backend/` root:

```env
DATABASE_URL=postgresql://user:password@localhost:5432/securedoc?ssl=false
GEMINI_API_KEY=your_google_ai_studio_key
JWT_SECRET=your_long_random_jwt_secret_32plus_chars
JWT_EXPIRES_IN=15m
REFRESH_TOKEN_SECRET=your_refresh_token_secret
REFRESH_EXPIRES_IN=7d
NODE_ENV=development
FRONTEND_URL=http://localhost:5173
```

---

## Security Properties

| Property | Implementation |
|---|---|
| Client private key never leaves browser | Browser-only, non-exportable |
| Server private key never in API responses | Only loaded to memory at startup |
| PRIVATE chunks never reach Gemini | Sensitivity column checked BEFORE decrypt |
| Ephemeral session keys | Fresh AES-GCM key per request AND per response |
| Full audit trail | Every decrypt event logged with user ID, doc ID, chunk IDs, query hash |
