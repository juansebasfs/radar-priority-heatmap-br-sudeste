# -*- coding: utf-8 -*-
"""
Gera um HTML interativo com:
- Heatmaps de Feridos/Mortos (por UF e combinados),
- Pontos de Radares Propostos com filtros (UF, Eficácia, Prioridade),
- Contador de radares visíveis,
- Popups com UF, Eficácia, Prioridade, BR, KM, ID.

Autor: (seu nome)
Curso: Ciência de Dados (PT-BR) — material didático.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap, MiniMap


# -----------------------------
# 1) Funções utilitárias
# -----------------------------
def normalizar_coord(valor: object) -> float:
    """
    Converte strings numéricas com vírgula decimal/sep. de milhares para float.
    Retorna np.nan se não conseguir converter.
    """
    if pd.isna(valor):
        return np.nan
    s = str(valor).strip().replace(",", ".")
    # remove ponto de milhares: 1.234.567,89 -> 1234567.89
    s = re.sub(r'(?<=\d)\.(?=\d{3}(\D|$))', "", s)
    # se ainda restou mais de um ponto, mantém apenas o primeiro
    if s.count(".") > 1:
        first = s.find(".")
        s = s[: first + 1] + s[first + 1 :].replace(".", "")
    try:
        return float(s)
    except Exception:
        return np.nan


def aggregate_for_heatmap(df: pd.DataFrame, weight_col: str, decimals: int = 3) -> pd.DataFrame:
    """
    Agrega pontos em 'grades' espaciais (arredondando lat/lon) para acelerar o heatmap.
    - decimals=3 ~ ~110 m (aprox); ajuste se quiser mais/menos suavização.
    """
    d = df.copy()
    d["lat_bin"] = d["latitude"].round(decimals)
    d["lon_bin"] = d["longitude"].round(decimals)
    g = d.groupby(["uf", "lat_bin", "lon_bin"], as_index=False)[weight_col].sum()
    g.rename(columns={"lat_bin": "lat", "lon_bin": "lon", weight_col: "weight"}, inplace=True)
    return g[g["weight"] > 0]


def ensure_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Garante que as colunas da lista existam no DataFrame; cria vazias (NA) se não existirem.
    Útil para campos opcionais que vão pro popup (br, trecho_km_final, id_trecho).
    """
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df


# -----------------------------
# 2) Pipeline principal
# -----------------------------
def gerar_html_mapa(
    csv_acidentes: Path,
    csv_radares: Path,
    arquivo_saida: Path,
):
    ufs_sudeste = {"SP", "MG", "RJ", "ES"}

    # ----- Ler acidentes
    df_acc = pd.read_csv(csv_acidentes, encoding="utf-8")
    df_acc.columns = [c.strip().lower() for c in df_acc.columns]
    # normaliza UF e filtra Sudeste
    df_acc["uf"] = df_acc["uf"].astype(str).str.upper().str.strip()
    df_acc = df_acc[df_acc["uf"].isin(ufs_sudeste)].copy()
    # coordenadas já devem estar limpas, mas reforçamos:
    df_acc["latitude"] = df_acc["latitude"].apply(normalizar_coord)
    df_acc["longitude"] = df_acc["longitude"].apply(normalizar_coord)
    df_acc = df_acc.dropna(subset=["latitude", "longitude"])

    # [Opcional] reforçar filtro de BR se quiser
    # br_permitidas = {40,50,101,116,135,146,153,251,259,262,267,354,356,364,365,381,393,447,452,459,460,465,474,488,493,495,601}
    # if "br" in df_acc.columns:
    #     df_acc["br"] = pd.to_numeric(df_acc["br"], errors="coerce")
    #     df_acc = df_acc[df_acc["br"].isin(br_permitidas)].copy()

    # ----- Ler radares
    df_rad = pd.read_csv(csv_radares, encoding="utf-8")
    df_rad.columns = [c.strip().lower() for c in df_rad.columns]

    # Detecta nomes de latitude/longitude (trecho_* ou direto)
    if "trecho_latitude_central" in df_rad.columns and "trecho_longitude_central" in df_rad.columns:
        df_rad = df_rad.rename(
            columns={
                "trecho_latitude_central": "latitude",
                "trecho_longitude_central": "longitude",
            }
        )
    # Garante colunas opcionais pro popup
    df_rad = ensure_columns(df_rad, ["br", "trecho_km_final", "id_trecho"])

    # Normaliza UF e filtra Sudeste
    df_rad["uf"] = df_rad["uf"].astype(str).str.upper().str.strip()
    df_rad = df_rad[df_rad["uf"].isin(ufs_sudeste)].copy()

    # Limpeza de coordenadas/índices
    df_rad["latitude"] = df_rad["latitude"].apply(normalizar_coord)
    df_rad["longitude"] = df_rad["longitude"].apply(normalizar_coord)
    for col in ["prob_alta_eficacia", "indice_prioridade_norm"]:
        df_rad[col] = pd.to_numeric(df_rad[col], errors="coerce")
    # Extras p/ popup
    if "br" in df_rad.columns:
        df_rad["br"] = pd.to_numeric(df_rad["br"], errors="coerce")
    if "trecho_km_final" in df_rad.columns:
        df_rad["trecho_km_final"] = pd.to_numeric(df_rad["trecho_km_final"], errors="coerce")
    if "id_trecho" in df_rad.columns:
        df_rad["id_trecho"] = df_rad["id_trecho"].astype(str)

    # Remove linhas inválidas
    df_rad = df_rad.dropna(subset=["latitude", "longitude", "prob_alta_eficacia", "indice_prioridade_norm"])
    df_rad["prob_alta_eficacia"] = df_rad["prob_alta_eficacia"].clip(0, 100)
    df_rad["indice_prioridade_norm"] = df_rad["indice_prioridade_norm"].clip(0, 100)

    # ----- Agregações para heatmap
    agg_feridos = aggregate_for_heatmap(df_acc, "feridos", decimals=3)
    agg_mortos = aggregate_for_heatmap(df_acc, "mortos", decimals=3)

    # ----- Cria o mapa centrado no Sudeste (bounds fixos)
    lat_min, lat_max = -25.8, -14.0
    lon_min, lon_max = -53.6, -39.0
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="CartoDB positron", control_scale=True)
    m.fit_bounds([[lat_min, lon_min], [lat_max, lon_max]])
    MiniMap(toggle_display=True).add_to(m)

    # ----- Funções auxiliares de camadas
    def add_heat_layers_by_uf(map_obj, df_agg, label_prefix, radius=12, blur=15, max_zoom=10):
        """
        Cria 4 camadas (SP, MG, ES, RJ) com heatmap somando os pesos por célula agregada.
        """
        for uf in ["SP", "MG", "ES", "RJ"]:
            sub = df_agg[df_agg["uf"] == uf]
            if sub.empty:
                continue
            pts = sub[["lat", "lon", "weight"]].values.tolist()
            fg = folium.FeatureGroup(name=f"{label_prefix} - {uf}", show=False)
            HeatMap(pts, radius=radius, blur=blur, max_zoom=max_zoom).add_to(fg)
            fg.add_to(map_obj)

    def add_combined_layer(map_obj, df_agg, layer_name, radius=12, blur=15, max_zoom=10):
        """
        Camada combinada (todos os estados) para uma métrica (Feridos/Mortos).
        """
        sub = df_agg.groupby(["lat", "lon"], as_index=False)["weight"].sum()
        pts = sub[["lat", "lon", "weight"]].values.tolist()
        fg = folium.FeatureGroup(name=layer_name, show=True)  # visível por padrão
        HeatMap(pts, radius=radius, blur=blur, max_zoom=max_zoom).add_to(fg)
        fg.add_to(map_obj)

    # ----- Adiciona camadas
    add_heat_layers_by_uf(m, agg_feridos, "Feridos")
    add_heat_layers_by_uf(m, agg_mortos, "Mortos")
    add_combined_layer(m, agg_feridos, "Feridos - Todos")
    add_combined_layer(m, agg_mortos, "Mortos - Todos")

    # ----- Camada de Radares + dados em JSON
    fg_radares = folium.FeatureGroup(name="Radares Propostos", show=True)
    fg_radares.add_to(m)

    cols_popup = ["latitude", "longitude", "uf", "prob_alta_eficacia", "indice_prioridade_norm", "br", "trecho_km_final", "id_trecho"]
    radar_records = df_rad[cols_popup].to_dict(orient="records")
    radar_json = json.dumps(radar_records, ensure_ascii=False)
    # Técnica segura: dados em <div> escondido, e JS puro injetado no final:
    json_div = '<div id="radar-data" style="display:none;">' + radar_json + "</div>"
    m.get_root().html.add_child(folium.Element(json_div))

    # ----- JS de UI + marcadores (sem <script>, para evitar conflitos de template)
    map_var_name = m.get_name()
    fg_var_name = fg_radares.get_name()

    js_code = f"""
(function() {{
  var MAP_VAR_NAME = '{map_var_name}';
  var FG_VAR_NAME  = '{fg_var_name}';
  var radarData = JSON.parse(document.getElementById('radar-data').textContent || '[]');
  var markers = [];

  function colorBlueToRed(v) {{
    // 0 -> azul, 100 -> vermelho
    var t = Math.max(0, Math.min(100, v)) / 100.0;
    var r = Math.round(255 * t);
    var g = 0;
    var b = Math.round(255 * (1 - t));
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  }}

  function getUFsSelecionadas() {{
    var ufs = [];
    document.querySelectorAll('.ufChk').forEach(function(c) {{ if (c.checked) ufs.push(c.value); }});
    return ufs;
  }}

  function clearMarkers(layer) {{
    markers.forEach(function(mk) {{ layer.removeLayer(mk); }});
    markers = [];
  }}

  function percentile(arr, p) {{
    if (!arr.length) return 0;
    var sorted = arr.slice().sort(function(a,b) {{ return a-b; }});
    var idx = Math.min(sorted.length-1, Math.max(0, Math.round((p/100) * (sorted.length-1))));
    return sorted[idx];
  }}

  function computePriorityCutoff(ufSel, mode, value) {{
    // 'min': corte fixo; 'top': pega percentil com base nas UFs selecionadas (top X% -> percentil 100-X)
    if (mode === 'min') return value;
    var vals = radarData.filter(function(pt) {{ return ufSel.indexOf(pt.uf) !== -1; }})
                        .map(function(pt) {{ return pt.indice_prioridade_norm; }});
    var perc = 100 - Math.max(0, Math.min(100, value));
    return percentile(vals, perc);
  }}

  function renderMarkers(mapObj, layer) {{
    clearMarkers(layer);
    var ufSel   = getUFsSelecionadas();

    var probMin = parseInt(document.getElementById('probSlider').value, 10) || 0;  // eficácia mínima (slice)
    var prioMode = document.querySelector('input[name="prioMode"]:checked').value; // 'min' | 'top'
    var prioVal;
    if (prioMode === 'min') {{
      prioVal = parseInt(document.getElementById('prioValueMin').value, 10) || 0;
      document.getElementById('prioSliderMinWrapper').style.display = 'block';
      document.getElementById('prioTopWrapper').style.display = 'none';
      document.getElementById('prioValMin').textContent = prioVal;
    }} else {{
      prioVal = parseInt(document.getElementById('prioValueTop').value, 10) || 20;
      document.getElementById('prioSliderMinWrapper').style.display = 'none';
      document.getElementById('prioTopWrapper').style.display = 'block';
      document.getElementById('prioValTop').textContent = prioVal;
    }}
    document.getElementById('probVal').textContent = probMin;

    var modeEls = document.getElementsByName('radarMode');
    var mode = 'off';
    for (var i=0;i<modeEls.length;i++) {{ if (modeEls[i].checked) mode = modeEls[i].value; }}

    if (mode === 'off') {{
      document.getElementById('radarCount').textContent = 0;
      return;
    }}

    var prioCutoff = computePriorityCutoff(ufSel, prioMode, prioVal);

    var visibleCount = 0;
    radarData.forEach(function(pt) {{
      if (ufSel.indexOf(pt.uf) === -1) return;
      if (pt.prob_alta_eficacia < probMin) return;                  // eficácia ≥ slider
      if (pt.indice_prioridade_norm < prioCutoff) return;           // prioridade ≥ corte (ou top%)

      visibleCount += 1;
      var metric = (mode === 'ef') ? pt.prob_alta_eficacia : pt.indice_prioridade_norm;
      var color = colorBlueToRed(metric);

      var brTxt = (pt.br === null || pt.br === undefined || String(pt.br) === 'nan') ? '' : String(pt.br);
      var kmTxt = (pt.trecho_km_final === null || pt.trecho_km_final === undefined) ? '' : String(pt.trecho_km_final);
      var idTxt = (pt.id_trecho === null || pt.id_trecho === undefined || String(pt.id_trecho) === 'nan') ? '' : String(pt.id_trecho);

      var html = ''
        + '<div style="font-size:12px;">'
        + '<b>Radar proposto</b><br/>'
        + 'UF: ' + pt.uf + '<br/>'
        + 'Eficácia: ' + pt.prob_alta_eficacia + '<br/>'
        + 'Prioridade: ' + pt.indice_prioridade_norm + '<br/>'
        + 'BR-' + brTxt + '<br/>'
        + 'KM: ' + kmTxt + '<br/>'
        + 'ID: ' + idTxt
        + '</div>';

      var marker = L.circleMarker([pt.latitude, pt.longitude], {{
        radius: 6,
        color: color,
        fillColor: color,
        fillOpacity: 0.85,
        weight: 1
      }}).bindPopup(html);

      marker.addTo(layer);
      markers.push(marker);
    }});

    // Atualiza contador de radares visíveis
    document.getElementById('radarCount').textContent = visibleCount;
  }}

  function setupUI(mapObj, layer) {{
    var control = L.control({{position: 'topright'}});
    control.onAdd = function(map) {{
      var div = L.DomUtil.create('div', 'leaflet-bar leaflet-control');
      div.style.background = 'white';
      div.style.padding = '10px';
      div.style.minWidth = '260px';
      div.style.maxWidth = '300px';
      div.style.boxShadow = '0 1px 5px rgba(0,0,0,0.4)';
      div.innerHTML = ''
        + '<h4 style="margin:0 0 8px 0;">Radares Propostos</h4>'
        + '<div style="font-size:12px;line-height:1.2;margin-bottom:6px;">'
        + '  <strong>Filtrar por UF</strong><br/>'
        + '  <label><input type="checkbox" class="ufChk" value="SP" checked> SP</label>'
        + '  <label><input type="checkbox" class="ufChk" value="RJ" checked> RJ</label><br/>'
        + '  <label><input type="checkbox" class="ufChk" value="ES" checked> ES</label>'
        + '  <label><input type="checkbox" class="ufChk" value="MG" checked> MG</label>'
        + '</div>'
        + '<div style="font-size:12px;margin-bottom:6px;">'
        + '  <label><strong>Índice de Eficácia</strong> (0–100):<br/>'
        + '  <input id="probSlider" type="range" min="0" max="100" step="1" value="70" style="width:100%;"></label>'
        + '  <div style="display:flex;justify-content:space-between;">'
        + '    <span>min: <span id="probVal">70</span></span>'
        + '    <span>radares: <strong id="radarCount">0</strong></span>'
        + '  </div>'
        + '</div>'
        + '<div style="font-size:12px;margin-bottom:6px;">'
        + '  <strong>Filtro de Prioridade</strong><br/>'
        + '  <label><input type="radio" name="prioMode" value="min" checked> Mínimo</label> '
        + '  <label><input type="radio" name="prioMode" value="top"> Top %</label>'
        + '  <div id="prioSliderMinWrapper" style="margin-top:6px;">'
        + '    <input id="prioValueMin" type="range" min="0" max="100" step="1" value="0" style="width:100%;">'
        + '    <div style="text-align:right;">min: <span id="prioValMin">0</span></div>'
        + '  </div>'
        + '  <div id="prioTopWrapper" style="display:none;margin-top:6px;">'
        + '    <input id="prioValueTop" type="range" min="1" max="50" step="1" value="20" style="width:100%;">'
        + '    <div style="text-align:right;">Top: <span id="prioValTop">20</span>%</div>'
        + '  </div>'
        + '</div>'
        + '<div style="font-size:12px;margin-bottom:6px;">'
        + '  <strong>Colorir pontos por</strong><br/>'
        + '  <label><input type="radio" name="radarMode" value="ef" checked> Eficácia</label> '
        + '  <label><input type="radio" name="radarMode" value="pr"> Prioridade</label> '
        + '  <label><input type="radio" name="radarMode" value="off"> Ocultar</label>'
        + '</div>'
        + '<button id="resetBtn" style="width:100%;padding:6px;margin-top:4px;">Limpar filtros</button>';
      L.DomEvent.disableScrollPropagation(div);
      L.DomEvent.disableClickPropagation(div);
      return div;
    }};
    control.addTo(mapObj);

    // Eventos
    document.addEventListener('input', function(e) {{
      if (!e.target) return;
      var id = e.target.id;
      if (id === 'probSlider' || id === 'prioValueMin' || id === 'prioValueTop' || e.target.classList.contains('ufChk') || e.target.name === 'radarMode' || e.target.name === 'prioMode') {{
        renderMarkers(mapObj, layer);
      }}
    }});
    document.addEventListener('click', function(e) {{
      if (e.target && e.target.id === 'resetBtn') {{
        document.querySelectorAll('.ufChk').forEach(function(c) {{ c.checked = true; }});
        document.querySelector('input[name="prioMode"][value="min"]').checked = true;
        document.getElementById('probSlider').value = 70;
        document.getElementById('prioValueMin').value = 0;
        document.getElementById('prioValueTop').value = 20;
        document.querySelector('input[name="radarMode"][value="ef"]').checked = true;
        renderMarkers(mapObj, layer);
      }}
    }});

    // Primeira renderização
    renderMarkers(mapObj, layer);
  }}

  // Espera até que o mapa/feature group existam como variáveis globais
  function waitReady() {{
    var mapObj = window[MAP_VAR_NAME];
    var layer  = window[FG_VAR_NAME];
    if (!mapObj || !layer) {{ return setTimeout(waitReady, 60); }}
    setupUI(mapObj, layer);
  }}
  waitReady();
}})();
"""
    m.get_root().script.add_child(folium.Element(js_code))

    # Controle de camadas
    for key in list(m._children.keys()):
        if "layer_control" in key:
            del m._children[key]
    folium.LayerControl(collapsed=False).add_to(m)

    # Salva HTML
    m.save(str(arquivo_saida))
    print(f"[OK] HTML gerado em: {arquivo_saida.resolve()}")


# -----------------------------
# 3) CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Gera HTML com heatmap de acidentes + radares propostos (Sudeste)")
    p.add_argument("--acidentes", required=True, type=Path, help="Caminho do CSV de acidentes (pré-processado)")
    p.add_argument("--radares", required=True, type=Path, help="Caminho do CSV de radares propostos")
    p.add_argument("--saida", default=Path("heatmap_acidentes_sudeste_com_radares.html"), type=Path, help="Arquivo HTML de saída")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gerar_html_mapa(args.acidentes, args.radares, args.saida)
