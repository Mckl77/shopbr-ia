-- ============================================================
-- Projet #6 - Big Data Certification - ShopBR
-- redshift_optimize.sql : Performances et monitoring
-- ============================================================

VACUUM FULL analytics.late_rate_by_state;
VACUUM FULL analytics.late_rate_by_category;
VACUUM FULL analytics.seller_performance;
VACUUM FULL analytics.cross_state_impact;
VACUUM FULL analytics.delay_review_impact;
VACUUM FULL analytics.monthly_trend;
VACUUM FULL analytics.weekday_pattern;

ANALYZE analytics.late_rate_by_state;
ANALYZE analytics.late_rate_by_category;
ANALYZE analytics.seller_performance;
ANALYZE analytics.cross_state_impact;
ANALYZE analytics.delay_review_impact;
ANALYZE analytics.monthly_trend;
ANALYZE analytics.weekday_pattern;

-- Verification distribution
SELECT schema, "table", diststyle, skew_rows, pct_used
FROM svv_table_info
WHERE schema = 'analytics'
ORDER BY skew_rows DESC;

-- Requetes lentes (24h)
SELECT userid, TRIM(querytxt) AS query_text, starttime,
       DATEDIFF(seconds, starttime, endtime) AS duration_sec
FROM stl_query
WHERE starttime >= DATEADD(hour,-24,CURRENT_TIMESTAMP) AND userid > 1
ORDER BY duration_sec DESC LIMIT 20;

-- ============================================================
-- Requetes metier de validation (donnees reelles attendues)
-- ============================================================

-- Q1 : Quel est le taux de retard global ?
-- Attendu : ~6.65% (vérifié sur le dataset source)
SELECT * FROM reporting.v_executive_dashboard;

-- Q2 : Impact du retard sur la note client
-- Attendu : ~2.27/5 si en retard vs ~4.29/5 si à l'heure
SELECT
    CASE WHEN is_late THEN 'En retard' ELSE 'A l''heure' END AS statut,
    total_orders,
    avg_review_score,
    bad_review_rate_pct
FROM analytics.delay_review_impact
ORDER BY is_late;

-- Q3 : Top 10 etats les plus a risque de retard
SELECT * FROM reporting.v_top_risk_states;

-- Q4 : Livraison inter-etats vs meme etat
-- Attendu : ~63.8% des commandes sont inter-etats
SELECT
    CASE WHEN cross_state_delivery THEN 'Inter-etats' ELSE 'Meme etat' END AS type_livraison,
    total_orders,
    late_rate_pct,
    avg_delivery_days
FROM analytics.cross_state_impact;

-- Q5 : Top 10 vendeurs les plus fiables (faible taux de retard, volume significatif)
SELECT main_seller_id, seller_state, total_orders, late_rate_pct, avg_review_score
FROM analytics.seller_performance
WHERE total_orders >= 50
ORDER BY late_rate_pct ASC
LIMIT 10;
