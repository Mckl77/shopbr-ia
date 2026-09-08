# Politique de Securite et Gestion des Acces
## Projet Big Data — ShopBR (Olist E-Commerce Bresil)
### Architecte en Intelligence Artificielle — Certification Bloc 6

---

## 1. Perimetre

Architecture couvrant : AWS S3, AWS Glue (PySpark serverless), Redshift Serverless, CloudWatch, CloudTrail.
Donnees : 99 441 commandes, 99 441 clients, 3 095 vendeurs (LGPD bresilienne applicable).

---

## 2. Roles IAM (Moindre privilege)

### GlueJobS3Role
```json
{ "Action": ["s3:GetObject","s3:PutObject","s3:DeleteObject","s3:ListBucket"],
  "Resource": ["arn:aws:s3:::votre-bucket-shopbr/*"], "Effect": "Allow" }
```

### RedshiftS3CopyRole
```json
{ "Action": ["s3:GetObject","s3:GetBucketLocation"],
  "Resource": ["arn:aws:s3:::votre-bucket-shopbr/gold/*"], "Effect": "Allow" }
```

### DataEngineerRole — Acces complet aux jobs Glue + lecture Redshift
### AnalystReadOnlyRole — Lecture seule schema reporting

---

## 3. Permissions Redshift

```sql
CREATE GROUP data_engineers;
CREATE GROUP analysts;

GRANT ALL ON SCHEMA analytics TO GROUP data_engineers;
GRANT ALL ON ALL TABLES IN SCHEMA analytics TO GROUP data_engineers;

GRANT USAGE ON SCHEMA analytics TO GROUP analysts;
GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO GROUP analysts;
GRANT USAGE ON SCHEMA reporting TO GROUP analysts;
GRANT SELECT ON ALL TABLES IN SCHEMA reporting TO GROUP analysts;
```

---

## 4. Chiffrement

| Ressource | Methode | Cle |
|---|---|---|
| S3 (toutes couches) | SSE-KMS | CMK `shopbr-key` |
| Redshift | AES-256 + KMS | CMK `redshift-key` |
| AWS Glue DBFS | SSE-KMS | Cle managee |
| Secrets Manager | KMS | Cle managee AWS |

Chiffrement en transit : HTTPS force sur S3, SSL Redshift, TLS 1.2+ AWS Glue.
Secrets jamais en clair : AWS Glue Secrets API + AWS Secrets Manager (rotation 90 jours).

---

## 5. Securite reseau

```
VPC sa-east-1 (region Bresil — proximite donnees + conformite LGPD)
  Subnet prive AWS Glue : 10.0.1.0/24
  Subnet prive Redshift   : 10.0.2.0/24
```
Redshift et AWS Glue en subnets prives, acces externe via VPN uniquement,
flux S3 via VPC Endpoint.

---

## 6. Conformite LGPD (Lei Geral de Protecao de Dados — Bresil)

Le dataset ShopBR contient des donnees personnelles de citoyens bresiliens
(customer_city, customer_state, zip code, identifiants uniques). La LGPD
s'applique de la meme maniere que le RGPD en Europe :

- Minimisation : seules les colonnes necessaires a la prediction de retard
  sont conservees en Gold (customer_id pseudonymise, pas de nom/email)
- Finalite : les donnees sont utilisees exclusivement pour l'amelioration
  du service de livraison, pas de revente a des tiers
- Droit d'acces : tout client peut demander la suppression de ses donnees
  (procedure documentee aupres du DPO ShopBR)
- Tracabilite : audit complet CloudTrail, retention 365 jours
- Anonymisation : customer_unique_id est deja un hash dans le dataset source

---

## 7. Monitoring et audit

| Alarme | Condition | Action |
|---|---|---|
| RedshiftHighCPU | CPU > 80% pendant 5 min | SNS -> Email |
| RedshiftConnectionFailed | > 10 echecs en 5 min | SNS + Slack |
| S3UnauthorizedAccess | AccessDenied sur bucket | Alerte immediate |
| AWS GlueClusterStopped | Arret inattendu | SNS -> Email |
| RedshiftDiskUsage | Disque > 85% | SNS + scaling |

---

## 8. Checklist de conformite

- [x] MFA active sur tous les comptes IAM humains
- [x] Chiffrement S3, Redshift, DBFS actif
- [x] CloudTrail actif (365 jours retention)
- [x] Redshift accessible uniquement depuis VPC prive
- [x] Secrets jamais en clair (AWS Glue Secrets + AWS SM)
- [x] LGPD : finalite documentee, droit a l'effacement procedure

---

*References : LGPD (Lei 13.709/2018), AWS IAM Best Practices, Amazon Redshift Security Guide*
