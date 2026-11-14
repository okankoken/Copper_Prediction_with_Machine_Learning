"""Read-only research view. The original Lead-Lag page remains independent."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st


LABELS = {'mining_com': 'Mining.com', 'chile_local': 'Şili yerel yayınları',
          'peru_local': 'Peru yerel yayınları', 'mining_company': 'Madencilik şirketleri',
          'financial_tone': 'Finansal haber tonu', 'copper_explicit': 'Başlığında bakır geçenler'}
BANDS = {'next_two_blocks': 'Sonraki iki dönem', 'second_month': 'İkinci ay', 'third_month': 'Üçüncü ay'}


def render_news_research(root):
    directory = Path(root) / 'data/processed/diagnostics/news_history_experiment'
    st.title('Bakır Haber ve Gecikme Laboratuvarı')
    st.caption('Mining.com, Şili ve Peru yerel haberleri ile madencilik şirketleri. Etkin aylık tahmin ve ayrı on günlük araştırma sonuçları.')
    active_path = Path(root) / 'data/predictions/copper_production_forecast_latest.csv'
    if active_path.exists():
        active = pd.read_csv(active_path)
        if 'production_policy_version' in active and set(active.production_policy_version).issubset({'monthly_news_joint_ml_v1','monthly_news_horizon_ml_v1','monthly_news_blocks_ml_v1'}):
            st.subheader('Etkin aylık haberli model — H1–H12')
            st.dataframe(active[['forecast_date','final_forecast_price_usd_per_ton','evaluation_type','confidence_flag']].rename(columns={
                'forecast_date':'Ay','final_forecast_price_usd_per_ton':'Aylık ortalama USD/ton',
                'evaluation_type':'Tür','confidence_flag':'Güven etiketi'}),hide_index=True,use_container_width=True)
            blocks_path = Path(root) / 'data/predictions/copper_production_blocks_latest.csv'
            if blocks_path.exists() and 'monthly_news_blocks_ml_v1' in set(active.production_policy_version):
                blocks = pd.read_csv(blocks_path)
                if set(blocks.generated_at) == set(active.generated_at):
                    month = st.selectbox('Üç dönemini görmek istediğin ay', active.forecast_date.tolist())
                    horizon = active.loc[active.forecast_date.eq(month), 'horizon'].iloc[0]
                    st.dataframe(blocks.loc[blocks.horizon.eq(horizon), ['period_start','period_end',
                        'price_usd_per_ton','trading_days','observed_days']], hide_index=True)
            st.caption('Etkin modelin dönemleri yukarıdaki ay seçicisinde görünür.')
            contribution_path = Path(root) / 'data/processed/diagnostics/monthly_news/current_feature_contributions.csv'
            if contribution_path.exists():
                with st.expander('Etkin aylık modeli hangi girdiler yönlendiriyor?'):
                    horizon = st.selectbox('Aylık tahmin ufku',list(range(1,13)),key='active_monthly_news_horizon')
                    inputs = pd.read_csv(contribution_path)
                    inputs = inputs[inputs.horizon.eq(horizon)].copy()
                    inputs['absolute_contribution'] = inputs.contribution_usd.abs()
                    st.dataframe(inputs.nlargest(20,'absolute_contribution')[['feature','family','contribution_usd']],hide_index=True,use_container_width=True)
                    st.caption('Eğitim ortalaması referansına göre USD katkısı; nedensel etki değildir. Negatif ve pozitif değerler birlikte tahmini oluşturur.')
    if not (directory / 'manifest.json').exists():
        st.info('Haber araştırması çıktıları henüz bu ortamda bulunmuyor.')
        return
    manifest = json.loads((directory / 'manifest.json').read_text())

    def read(name):
        path = directory / name
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    columns = st.columns(4)
    columns[0].metric('Ayıklanmış başlık', f'{manifest["headline_records"]:,}')
    columns[1].metric('İlk kayıt', manifest['first_headline'][:10])
    columns[2].metric('Karar tarihi', manifest['decision_date'])
    columns[3].metric('Son fiyat kaydı', manifest['price_asof'])
    st.info('Arşiv kapsamı zaman içinde değişir. Toplanan başlık sayısı toplam haber hacmi değildir. Haber bulunmayan dönemler nötr duygu olarak doldurulmaz.')
    counts = pd.DataFrame([{'Kaynak grubu': LABELS.get(k, k), 'Başlık': v} for k, v in manifest['source_counts'].items()])
    st.dataframe(counts, hide_index=True, use_container_width=True)
    latest = read('latest_headline_review.csv')
    if not latest.empty:
        with st.expander('Toplanan Mining.com, Şili ve Peru haberlerini gör'):
            source = st.selectbox('Yayın grubu', list(latest.channel.unique()), format_func=lambda k: LABELS.get(k, k), key='news_headline_source')
            selected_headlines = latest[latest.channel.eq(source)].sort_values('published_at', ascending=False)
            st.dataframe(selected_headlines[['published_at', 'title', 'publisher', 'tone_net', 'url']],
                hide_index=True, use_container_width=True)
            st.caption('Ton, finansal metnin genel olumlu/olumsuz skorudur; bakır fiyatının yönü değildir. Başlık tarihi ve tam metin bağlamı ayrıca kontrol edilmelidir.')

    st.subheader('Haberli ve habersiz on günlük tahmin')
    predictions = read('backtest_predictions.csv')
    if not predictions.empty:
        current = predictions[predictions['asof'].eq(predictions['asof'].max())]
        table = current.pivot(index=['period_start', 'period_end'], columns='model', values='prediction').reset_index()
        selected = manifest['best_news_candidate_on_2025']
        table['Haber eklenince fark'] = table[selected] - table['baseline_pls_5y']
        display = table[['period_start', 'period_end', 'baseline_pls_5y', selected, 'Haber eklenince fark']].rename(columns={
            'period_start': 'Başlangıç', 'period_end': 'Bitiş', 'baseline_pls_5y': 'Habersiz USD/ton', selected: 'Haberli USD/ton'})
        st.dataframe(display.style.format({c: '{:,.1f}' for c in display.select_dtypes('number')}), hide_index=True, use_container_width=True)
        chart = table.set_index('period_start')[['baseline_pls_5y', selected]].rename(columns={'baseline_pls_5y': 'Habersiz', selected: 'Haberli'})
        st.line_chart(chart)
    st.caption('1–10, 11–20 ve 21–ay sonu işlem günü ortalamalarıdır. İlk dönem, bilinen fiyatlarla henüz raporlanmamış günlerin tahminini birleştirir. Fark, model yeniden eğitildiğinde oluşan toplam farktır; haberin nedensel fiyat etkisi değildir.')

    st.subheader('Haberler geçmiş testte hata azalttı mı?')
    evidence = read('news_incremental_evidence.csv')
    if not evidence.empty:
        evidence = evidence[evidence.model.eq(manifest['best_news_candidate_on_2025'])].copy()
        evidence['band'] = evidence.band.map(BANDS)
        st.dataframe(evidence[['band', 'monthly_blocks', 'mean_gain_usd', 'bootstrap_low', 'bootstrap_high']].rename(columns={
            'band': 'Ufuk', 'monthly_blocks': 'Ay kümesi', 'mean_gain_usd': 'Ortalama hata azalması USD',
            'bootstrap_low': '%95 aralık altı', 'bootstrap_high': '%95 aralık üstü'}), hide_index=True, use_container_width=True)
    st.caption('Pozitif değer haberli modelin daha düşük mutlak hata verdiğini gösterir. Aralık sıfırı içeriyorsa katkı belirsizdir. Aylık kümeli bootstrap aralıkları keşifseldir; çoklu denemeler ve az sayıda ay nedeniyle kesin kanıt değildir.')
    ablations = read('source_ablation_evidence.csv')
    if not ablations.empty:
        with st.expander('Mining.com, Şili, Peru ve haber tonu ayrı katkıları'):
            ablations['removed'] = ablations.removed.map(LABELS)
            ablations['band'] = ablations.band.map(BANDS)
            st.dataframe(ablations.rename(columns={'removed': 'Çıkarılan kaynak', 'band': 'Ufuk',
                'mean_gain_usd': 'Kaynağı korumanın hata kazancı USD'}), hide_index=True, use_container_width=True)

    st.subheader('Hangi haber göstergesi kaç ay önden ilişkili?')
    st.caption('2025 öncesinde hesaplanan Spearman ilişkileri. Hedef, seçilen ileriki ayın aylık bakır getirisidir. İlişki, nedensellik veya tek başına tahmin başarısı anlamına gelmez.')
    lags = read('news_lead_lag_all_tests.csv')
    if not lags.empty:
        left, middle, right = st.columns(3)
        lead = left.slider('En fazla ileri ay', 1, 12, 6, key='news_max_lead')
        minimum = middle.slider('En az gözlem', 36, 110, 60, key='news_minimum')
        group = right.selectbox('Haber grubu', ['Tümü'] + list(LABELS.values())[:4] + [LABELS['copper_explicit']], key='news_group')
        filtered = lags[(lags.best_lead_months <= lead) & (lags.observations >= minimum)].copy()
        if group != 'Tümü':
            key = next(k for k, v in LABELS.items() if v == group)
            filtered = filtered[filtered.indicator.str.startswith('news_' + key + '_')]
        filtered = filtered.sort_values('abs_correlation', ascending=False).drop_duplicates('indicator')
        reliable = int((lags.fdr_q < .05).sum())
        if reliable == 0:
            st.warning('Çoklu karşılaştırma düzeltmesinden sonra %5 eşiğini geçen tekil haber gecikmesi bulunmadı. Aşağıdaki sıralama araştırma ipucudur.')
        st.dataframe(filtered[['indicator', 'best_lead_months', 'lead_correlation', 'same_month_correlation', 'lead_advantage', 'observations', 'fdr_q']].head(30).rename(columns={
            'indicator': 'Gösterge', 'best_lead_months': 'İleri ay', 'lead_correlation': 'İleri ilişki',
            'same_month_correlation': 'Aynı ay ilişkisi', 'lead_advantage': 'İlişki avantajı', 'observations': 'Gözlem', 'fdr_q': 'FDR q'}),
            hide_index=True, use_container_width=True)
        st.download_button('Tüm gecikme testlerini indir', lags.to_csv(index=False), 'news_lead_lag.csv', mime='text/csv')

    contributions = read('current_feature_contributions.csv')
    if not contributions.empty:
        st.subheader('Güncel tahmini yukarı ve aşağı çeken haber girdileri')
        offset = st.selectbox('On günlük hedef sırası', list(range(1, 10)), index=1, key='news_contribution_offset')
        current = contributions[(contributions.offset == offset) & contributions.family.eq('news')]
        left, right = st.columns(2)
        with left:
            st.caption('Yukarı yönlü katkı, USD/ton')
            st.dataframe(current[current.contribution_usd > 0].nlargest(10, 'contribution_usd')[['feature', 'contribution_usd']], hide_index=True, use_container_width=True)
        with right:
            st.caption('Aşağı yönlü katkı, USD/ton')
            st.dataframe(current[current.contribution_usd < 0].nsmallest(10, 'contribution_usd')[['feature', 'contribution_usd']], hide_index=True, use_container_width=True)
        st.caption('Katkılar, aynı modelin eğitim ortalaması referansına göre hesaplanır ve tahmini yeniden oluşturacak şekilde doğrulanır. Kaynak çıkarma testiyle aynı ölçü değildir.')
    with st.expander('Finansal ton modeli ve araştırmanın sınırları'):
        st.dataframe(read('tone_classifier_validation.csv'), hide_index=True)
        st.write('İngilizce ve İspanyolca metinler, etiketli Financial PhraseBank cümleleriyle eğitilen bir sınıflandırıcıdan geçer. Sayısal ton = olumlu olasılık − olumsuz olasılık. İspanyolca eğitim metinleri makine çevirisidir; yerel haberlerde ayrıca insan kontrolü gerekir.')
        st.write(manifest['limitations'])
