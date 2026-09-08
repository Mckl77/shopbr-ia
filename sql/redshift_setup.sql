-- ============================================================
-- Projet #6 - Big Data Certification - ShopBR
-- redshift_setup.sql : Creation de l'entrepot (Serverless) et des tables
-- ============================================================
-- ARCHITECTURE v2 (proportionnee, FinOps/GreenOps) : on utilise
-- Redshift SERVERLESS et non un cluster provisionne. Facturation a
-- l'usage (RPU-heures, pause automatique hors requetes) : adapte au
-- volume reel (~120 Mo) sans serveur permanent.
-- Creation du namespace + workgroup serverless (a executer une fois) :

-- aws redshift-serverless create-namespace \
--   --namespace-name shopbr-ns \
--   --db-name shopbr_dw \
--   --admin-username admin \
--   --admin-user-password "<MOT_DE_PASSE>" \
--   --kms-key-id arn:aws:kms:sa-east-1:xxxx:key/xxxx \
--   --region sa-east-1

-- aws redshift-serverless create-workgroup \
--   --workgroup-name shopbr-wg \
--   --namespace-name shopbr-ns \
--   --base-capacity 4 \
--   --region sa-east-1
-- (base 4 RPU = configuration minimale, suffisante pour ce volume ;
--  montee en charge automatique si besoin, plafonnable via max-capacity)

-- Note : un cluster provisionne (ra3/dc2) a ete etudie puis ecarte
-- comme surdimensionne — cf cadrage v2 et estimation comparative
-- (~30 $/mois serverless vs ~2 700 $/mois permanent).

CREATE SCHEMA IF NOT EXISTS analytics AUTHORIZATION admin;
CREATE SCHEMA IF NOT EXISTS reporting AUTHORIZATION admin;

-- Table 1 : Taux de retard par etat (carte de chaleur du risque)
CREATE TABLE IF NOT EXISTS analytics.late_rate_by_state (
    customer_state      VARCHAR(2)     NOT NULL,
    total_orders        INTEGER        NOT NULL,
    late_orders         INTEGER        NOT NULL,
    avg_delivery_days   DECIMAL(6,2),
    avg_review_score    DECIMAL(3,2),
    late_rate_pct       DECIMAL(5,2),
    inserted_at         TIMESTAMP      DEFAULT CURRENT_TIMESTAMP
)
DISTSTYLE ALL
SORTKEY(late_rate_pct);

-- Table 2 : Taux de retard par categorie produit
CREATE TABLE IF NOT EXISTS analytics.late_rate_by_category (
    product_category_name_english  VARCHAR(50)    NOT NULL,
    total_orders                   INTEGER        NOT NULL,
    late_orders                    INTEGER        NOT NULL,
    avg_price                      DECIMAL(8,2),
    avg_review_score                DECIMAL(3,2),
    late_rate_pct                  DECIMAL(5,2),
    inserted_at                    TIMESTAMP      DEFAULT CURRENT_TIMESTAMP
)
DISTSTYLE ALL
SORTKEY(late_rate_pct);

-- Table 3 : Performance des 50 plus gros vendeurs
CREATE TABLE IF NOT EXISTS analytics.seller_performance (
    main_seller_id      VARCHAR(64)    NOT NULL,
    seller_state         VARCHAR(2),
    total_orders          INTEGER       NOT NULL,
    late_orders            INTEGER      NOT NULL,
    avg_delivery_days       DECIMAL(6,2),
    avg_review_score         DECIMAL(3,2),
    avg_order_value           DECIMAL(8,2),
    late_rate_pct              DECIMAL(5,2),
    seller_rank                  SMALLINT NOT NULL,
    inserted_at                   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
DISTKEY(main_seller_id)
SORTKEY(seller_rank);

-- Table 4 : Impact livraison inter-etats (2 lignes : true/false)
CREATE TABLE IF NOT EXISTS analytics.cross_state_impact (
    cross_state_delivery   BOOLEAN       NOT NULL,
    total_orders            INTEGER      NOT NULL,
    late_orders               INTEGER    NOT NULL,
    avg_delivery_days          DECIMAL(6,2),
    avg_review_score             DECIMAL(3,2),
    late_rate_pct                  DECIMAL(5,2),
    inserted_at                      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
DISTSTYLE ALL
SORTKEY(cross_state_delivery);

-- Table 5 : Impact du retard sur la note client (preuve du ROI projet)
CREATE TABLE IF NOT EXISTS analytics.delay_review_impact (
    is_late                BOOLEAN       NOT NULL,
    total_orders             INTEGER     NOT NULL,
    avg_review_score           DECIMAL(3,2),
    bad_reviews                   INTEGER,
    bad_review_rate_pct             DECIMAL(5,2),
    inserted_at                       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
DISTSTYLE ALL
SORTKEY(is_late);

-- Table 6 : Tendance mensuelle
CREATE TABLE IF NOT EXISTS analytics.monthly_trend (
    purchase_year       SMALLINT       NOT NULL,
    purchase_month        SMALLINT     NOT NULL,
    total_orders            INTEGER    NOT NULL,
    late_orders               INTEGER  NOT NULL,
    avg_order_value             DECIMAL(8,2),
    avg_review_score              DECIMAL(3,2),
    late_rate_pct                   DECIMAL(5,2),
    inserted_at                       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
DISTSTYLE ALL
SORTKEY(purchase_year, purchase_month);

-- Table 7 : Pattern par jour de semaine
CREATE TABLE IF NOT EXISTS analytics.weekday_pattern (
    purchase_weekday     SMALLINT      NOT NULL,
    total_orders            INTEGER    NOT NULL,
    late_orders                INTEGER NOT NULL,
    avg_delivery_days            DECIMAL(6,2),
    late_rate_pct                  DECIMAL(5,2),
    inserted_at                      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
DISTSTYLE ALL
SORTKEY(purchase_weekday);

-- Vue reporting : tableau de bord executif
CREATE OR REPLACE VIEW reporting.v_executive_dashboard AS
SELECT
    SUM(total_orders)                                  AS total_orders_2018,
    SUM(late_orders)                                    AS total_late_orders,
    ROUND(SUM(late_orders)::DECIMAL / NULLIF(SUM(total_orders),0) * 100, 2) AS global_late_rate_pct,
    AVG(avg_review_score)                                  AS avg_review_score,
    AVG(avg_order_value)                                      AS avg_order_value
FROM analytics.monthly_trend;

-- Vue reporting : top 10 etats a risque
CREATE OR REPLACE VIEW reporting.v_top_risk_states AS
SELECT customer_state, total_orders, late_rate_pct, avg_review_score
FROM analytics.late_rate_by_state
ORDER BY late_rate_pct DESC
LIMIT 10;

COMMENT ON TABLE analytics.delay_review_impact IS
'Table cle du projet : demontre le lien retard -> mauvaise note client. Source Olist.';
COMMENT ON TABLE analytics.seller_performance IS
'Top 50 vendeurs par volume, avec taux de retard et note moyenne associes.';
