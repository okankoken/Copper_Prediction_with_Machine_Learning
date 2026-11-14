from pathlib import Path

import pandas as pd
import streamlit as st


def render_data_inventory(root: Path, master: pd.DataFrame):
    st.title("Veri Envanteri")
    plan_path = root / "config/master_feature_plan.csv"
    plan = pd.read_csv(plan_path) if plan_path.exists() else pd.DataFrame()
    st.caption("Kaynak planı, aylık ana tablo ve tahmin girdileri")
    inventory, values, features = st.tabs(["Kaynak planı", "Aylık veriler", "Model girdileri"])

    with inventory:
        if plan.empty:
            st.info("Kaynak planı bu ortamda bulunamadı.")
        else:
            plan["aylik_tabloda"] = plan.output_column.isin(master.columns)
            group = st.selectbox("Kaynak grubu", ["Tümü", *sorted(plan.category.dropna().unique())])
            visible = plan if group == "Tümü" else plan[plan.category.eq(group)]
            st.dataframe(visible, hide_index=True)

    with values:
        if master.empty:
            st.info("Aylık ana tablo bu ortamda bulunamadı.")
        else:
            st.caption(f"{len(master)} aylık satır · {len(master.columns)} sütun")
            columns = st.multiselect("İncelenecek sütunlar", [c for c in master if c != "date"],
                                     placeholder="Tüm sütunlar gösteriliyor")
            selected = master[[c for c in ["date", *columns] if c in master]] if columns else master
            st.dataframe(selected, hide_index=True)
            st.download_button("Aylık ana tablonun tamamını indir", master.to_csv(index=False),
                               "copper_monthly_master.csv", "text/csv")

    with features:
        st.caption("Piyasa serilerinde 1, 3, 6 ve 12 ay. Diğer aylık serilerde 2, 3, 6 ve 12 ay. "
                   "Haber özetlerinde 0, 1 ve 3 ay. Yayın tarihi doğrulanmamış yıllık seriler kaynak planında görülebilir.")
        path = root / "data/processed/diagnostics/monthly_news/feature_dictionary.csv"
        if path.exists():
            dictionary = pd.read_csv(path)
            st.caption("Tahmin için üretilen aday girdiler ve dönüşümleri")
            st.dataframe(dictionary, hide_index=True)
        else:
            st.info("Model girdi sözlüğü bu ortamda bulunamadı.")
