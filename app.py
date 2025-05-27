# Plotter Streamlit based Web App developed as a Thesis Project by Assem Agabekova, KIMEP University.

import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import time
import seaborn as sns
from streamlit_lottie import st_lottie
from io import BytesIO


# loading image for the Home page
im = "https://phd.pp.ua/wp-content/uploads/2019/07/x3-730x410.png"

# website Ddsign
st.set_page_config(page_title="Plotter",page_icon = im,layout="wide")

# column profiling + eligibility for charts
def get_eligible_columns_from_profile(profile, df, plot_type):
    numeric = profile.get('numeric', [])
    categorical = profile.get('categorical', [])
    datetime = profile.get('datetime', [])

    # only apply uniqueness filtering to categorical/datetime
    def filter_categorical(cols, min_u=2, max_u=25):
        return [col for col in cols if min_u <= df[col].nunique() <= max_u]

    result = {"x": [], "y": [], "hue": []}

    if plot_type in ['Simple Bar Chart', 'Horizontal Bar Chart']:
        result["x"] = filter_categorical(categorical + datetime)
        result["y"] = numeric

    elif plot_type == 'Stacked Bar Chart':
        result["x"] = filter_categorical(categorical)
        result["y"] = numeric
        result["hue"] = filter_categorical(categorical)

    elif plot_type == 'Multiple Bar Chart':
        result["x"] = filter_categorical(categorical)
        result["y"] = numeric
        result["hue"] = [col for col in filter_categorical(categorical) if df[col].nunique() <= 3]

    elif plot_type == 'Pie Chart':
        result["x"] = filter_categorical(categorical)

    elif plot_type == 'Nested Pie Chart':
        # all categorical columns with 2–25 uniques
        eligible = filter_categorical(categorical)
        result["x"] = eligible
        # inner ring must be any other eligible categorical
        result["hue"] = [col for col in eligible]

    elif plot_type == 'Line Chart':
        result["x"] = filter_categorical(categorical + datetime)
        result["y"] = numeric
        result["hue"] = filter_categorical(categorical)

    elif plot_type == 'Scatter Plot':
        result["x"] = numeric
        result["y"] = numeric
        result["hue"] = filter_categorical(categorical)

    elif plot_type == 'Histogram':
        result["x"] = numeric
        result["hue"] = filter_categorical(categorical)

    return result

# all chart types definitions: bar charts (simple, horizontal, stacked, multiple), pie chart (simple, nested), histogram, line chart, scatter plot
# SIMPLE BAR CHART
def create_simple_bar_chart(
    df,
    x_column,
    y_column,
    y_agg_func='mean',
    aggregate_y=True,
    filter_column=None,
    filter_value=None,
    sort_order=None,
    show_values=True
):
    # apply filtering if selected
    if filter_column and filter_value and filter_value != 'None':
        df = df[df[filter_column] == filter_value]

    # aggregate
    y_agg_func = y_agg_func or "mean"
    grouped_df = df.groupby(x_column)[y_column].agg(y_agg_func)

    # sorting
    if sort_order == 'Ascending':
        grouped_df = grouped_df.sort_values(ascending=True)
    elif sort_order == 'Descending':
        grouped_df = grouped_df.sort_values(ascending=False)

    # label pieces
    prettify = lambda s: s.replace("_", " ").capitalize()
    y_agg_func = y_agg_func or "mean"  # default fallback
    agg_label = y_agg_func.capitalize()

    ylabel = f"{agg_label} of {prettify(y_column)}"

    xlabel = prettify(x_column)
    filter_text = ""
    if filter_column and filter_value and filter_value != 'None':
        filter_text = f" – for '{filter_value}' category in {prettify(filter_column)}"

    # plotting
    fig, ax = plt.subplots()
    grouped_df.plot(kind='bar', ax=ax, color='skyblue')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("")  # No title at the top
    plt.xticks(rotation=45, ha='right')

    if show_values:
        for i, v in enumerate(grouped_df.values):
            ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

    # chart display
    st.pyplot(fig)
    # saving button
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="Download as PNG",
        data=buf,
        file_name="simple_bar.png",
        mime="image/png"
    )


# HORIZONTAL BAR CHART
def create_horizontal_bar_chart(
    df,
    x_column,
    y_column,
    y_agg_func='mean',
    aggregate_y=True,
    filter_column=None,
    filter_value=None,
    sort_order=None,
    show_values=True,
    custom_x_label=None,
    custom_y_label=None
):
    # filtering
    if filter_column and filter_value and filter_value != 'None':
        df = df[df[filter_column] == filter_value]

    # aggregation
    y_agg_func = y_agg_func or "mean"
    grouped_df = df.groupby(x_column)[y_column].agg(y_agg_func)

    # sorting
    if sort_order == 'Ascending':
        grouped_df = grouped_df.sort_values(ascending=True)
    elif sort_order == 'Descending':
        grouped_df = grouped_df.sort_values(ascending=False)

    # label building
    prettify = lambda s: s.replace("_", " ").capitalize()
    xlabel = custom_x_label if custom_x_label else f"{y_agg_func.capitalize()} of {prettify(y_column)}"
    ylabel = custom_y_label if custom_y_label else prettify(x_column)

    filter_text = ""
    if filter_column and filter_value and filter_value != 'None':
        filter_text = f" – for '{filter_value}' in {prettify(filter_column)}"

    # plotting
    fig, ax = plt.subplots()
    grouped_df.plot(kind='barh', ax=ax, color='salmon')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("")  # Remove top title

    # value annotations
    if show_values:
        max_val = grouped_df.max()  # for offset calculation
        offset = max_val * 0.02  # 2% of max
        for i, v in enumerate(grouped_df.values):
            ax.text(
                v + offset,  # push past the end of the bar
                i,  # at the bar’s center vertically
                f'{v:.2f}',  # label
                va='center',  # vertical alignment
                ha='center',  # horizontal alignment
                rotation=90,  # rotate text vertical
                fontsize=9
            )

    # display
    st.pyplot(fig)
    # saving
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="Download as PNG",
        data=buf,
        file_name="horizontal_bar.png",
        mime="image/png"
    )

# MULTIPLE BAR CHART
def create_multiple_bar_chart(
    df,
    x_column,
    y_column,
    hue_column,
    y_agg_func='mean',
    filter_column=None,
    filter_value=None,
    show_values=True,
    custom_x_label=None,
    custom_y_label=None
):
    # validate required hue
    if not hue_column or hue_column == 'None':
        st.error("This chart requires a grouping column (Hue). Please select one.")
        return

    # filtering
    if filter_column and filter_value and filter_value != 'None':
        df = df[df[filter_column] == filter_value]

    # aggregation
    y_agg_func = y_agg_func or "mean"
    grouped_df = df.groupby([x_column, hue_column])[y_column].agg(y_agg_func).reset_index()

    # labels
    prettify = lambda s: s.replace("_", " ").capitalize()
    agg_label = y_agg_func.capitalize()
    xlabel = custom_x_label if custom_x_label else prettify(x_column)
    ylabel = custom_y_label if custom_y_label else f"{agg_label} of {prettify(y_column)}"

    filter_text = ""
    if filter_column and filter_value and filter_value != 'None':
        filter_text = f" – for '{filter_value}' category in {prettify(filter_column)}"

    # plotting
    sns.barplot(data=grouped_df, x=x_column, y=y_column, hue=hue_column, ax=ax)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=grouped_df, x=x_column, y=y_column, hue=hue_column, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("")
    plt.xticks(rotation=45, ha='right')

    if show_values:
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", label_type="edge", padding=2, fontsize=9)

    # move legend box outside to the right
    ax.legend(title=prettify(hue_column),
              bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()

    # chart display
    st.pyplot(fig)
    # download
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="Download as PNG",
        data=buf,
        file_name="multiple_bar.png",
        mime="image/png"
    )

# STACKED BAR CHART
def create_stacked_bar_chart(
    df,
    x_column,
    y_column,
    hue_column,
    y_agg_func='sum',
    filter_column=None,
    filter_value=None,
    show_values=True,
    custom_x_label=None,
    custom_y_label=None
):
    # validate required hue
    if not hue_column or hue_column == 'None':
        st.error("This chart requires a grouping column (Hue). Please select one.")
        return

    # filtering
    if filter_column and filter_value and filter_value != 'None':
        df = df[df[filter_column] == filter_value]

    # aggregation
    y_agg_func = y_agg_func or "mean"
    grouped_df = df.groupby([x_column, hue_column])[y_column].agg(y_agg_func).unstack(fill_value=0)

    # labels
    prettify = lambda s: s.replace("_", " ").capitalize()
    agg_label = y_agg_func.capitalize()
    xlabel = custom_x_label if custom_x_label else prettify(x_column)
    ylabel = custom_y_label if custom_y_label else f"{agg_label} of {prettify(y_column)}"

    filter_text = ""
    if filter_column and filter_value and filter_value != 'None':
        filter_text = f" – for '{filter_value}' category in {prettify(filter_column)}"

    # plot with room for legend
    fig, ax = plt.subplots(figsize=(8, 6))
    grouped_df.plot(kind='bar', stacked=True, ax=ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("")
    plt.xticks(rotation=45, ha='right')

    if show_values:
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", label_type="center", fontsize=8, color='white')

    # move legend out
    ax.legend(title=prettify(hue_column),
              bbox_to_anchor=(1.02, 1), loc='upper left')

    fig.tight_layout()
    # chart display
    st.pyplot(fig)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="Download as PNG",
        data=buf,
        file_name="stacked_bar.png",
        mime="image/png"
    )

# LINE CHART
def create_line_chart(
        df,
        x_column,
        y_column,
        y_agg_func='mean',
        aggregate_y=True,
        hue_column=None,
        filter_column=None,
        filter_value=None,
        show_values=False,
        custom_x_label=None,
        custom_y_label=None
):
    # filtering
    if filter_column and filter_value and filter_value != 'None':
        df = df[df[filter_column] == filter_value]
    # aggregation
    y_agg_func = y_agg_func or "mean"
    prettify = lambda s: s.replace("_", " ").capitalize()
    agg_label = y_agg_func.capitalize()

    if hue_column and hue_column != 'None':
        grouped_df = df.groupby([x_column, hue_column])[y_column].agg(y_agg_func).reset_index()
    else:
        grouped_df = df.groupby(x_column)[y_column].agg(y_agg_func).reset_index()

    # label setup
    xlabel = custom_x_label if custom_x_label else prettify(x_column)
    ylabel = custom_y_label if custom_y_label else f"{agg_label} of {prettify(y_column)}"

    filter_text = ""
    if filter_column and filter_value and filter_value != 'None':
        filter_text = f" – for '{filter_value}' category in {prettify(filter_column)}"

    # plotting
    fig, ax = plt.subplots()
    if hue_column and hue_column != 'None':
        for key, group in grouped_df.groupby(hue_column):
            ax.plot(group[x_column], group[y_column], label=str(key))
        ax.legend(title=prettify(hue_column))
    else:
        ax.plot(grouped_df[x_column], grouped_df[y_column], marker='o')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("")  # No top title
    plt.xticks(rotation=45, ha='right')

    # optional point values
    if show_values:
        if hue_column and hue_column != 'None':
            for _, group in grouped_df.groupby(hue_column):
                for x, y in zip(group[x_column], group[y_column]):
                    ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)
        else:
            for x, y in zip(grouped_df[x_column], grouped_df[y_column]):
                ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

    # chart display
    st.pyplot(fig)
    # download
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="Download as PNG",
        data=buf,
        file_name="line_chart.png",
        mime="image/png"
    )

# PIE CHART
def create_pie_chart(
    df,
    x_column,
    filter_column=None,
    filter_value=None,
    custom_x_label=None
):
    # filter
    if filter_column and filter_value not in (None, 'None'):
        df = df[df[filter_column] == filter_value]

    # prepare data
    counts = df[x_column].value_counts()
    labels = counts.index.tolist()
    sizes  = counts.values

    # palette
    cmap = sns.color_palette("tab20", len(sizes))

    # plotting
    fig, ax = plt.subplots()
    wedges, _ = ax.pie(
        sizes,
        labels=None,                    # no direct labels
        colors=cmap,
        startangle=90,
        wedgeprops=dict(width=0.5, edgecolor='w'),
        autopct=None                   # we'll put pct in legend
    )
    ax.set(aspect="equal")

    # egend entries with both label + pct
    legend_labels = [
        f"{lab}: {size/sizes.sum()*100:.1f}%"
        for lab, size in zip(labels, sizes)
    ]
    ax.legend(
        wedges,
        legend_labels,
        title=custom_x_label or x_column.replace("_", " ").title(),
        loc="center left",
        bbox_to_anchor=(1, 0, 0.3, 1)
    )

    # optional center text
    ax.text(0, 0, custom_x_label or "", ha='center', va='center', fontsize=14)
    # chart display
    st.pyplot(fig)
    # download
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="Download as PNG",
        data=buf,
        file_name="pie_chart.png",
        mime="image/png"
    )

# NESTED PIE CHART
def create_nested_pie_chart(
    df,
    outer_col,
    inner_col,
    filter_column=None,
    filter_value=None,
    custom_outer_label=None,
    custom_inner_label=None
):
    from matplotlib.patches import ConnectionPatch

    if filter_column and filter_value not in (None, 'None'):
        df = df[df[filter_column] == filter_value]

    # counts
    outer_counts = df[outer_col].value_counts()
    inner_counts = df.groupby(outer_col)[inner_col].value_counts()

    outer_labels = outer_counts.index.tolist()
    outer_sizes  = outer_counts.values
    inner_labels = []
    inner_sizes  = []
    for o in outer_labels:
        sub = inner_counts[o]
        inner_labels += [f"{o} – {i}" for i in sub.index]
        inner_sizes  += sub.values.tolist()

    # palettes
    outer_cmap = sns.color_palette("Set2", len(outer_sizes))
    inner_cmap = sns.color_palette("Pastel1", len(inner_sizes))

    fig, ax = plt.subplots()
    size = 0.3

    # outer ring
    wedges1, _ = ax.pie(
        outer_sizes,
        radius=1,
        labels=None,
        colors=outer_cmap,
        startangle=90,
        wedgeprops=dict(width=size, edgecolor='w')
    )
    # inner ring
    wedges2, _ = ax.pie(
        inner_sizes,
        radius=1 - size,
        labels=None,
        colors=inner_cmap,
        startangle=90,
        wedgeprops=dict(width=size, edgecolor='w')
    )

    ax.set(aspect="equal")

    # legend: combine both rings
    legend_entries = wedges1 + wedges2
    legend_labels  = [
        f"{o}: {s/outer_sizes.sum()*100:.1f}%"
        for o, s in zip(outer_labels, outer_sizes)
    ] + [
        f"{l}: {v/sum(inner_sizes)*100:.1f}%"
        for l, v in zip(inner_labels, inner_sizes)
    ]

    ax.legend(
        legend_entries,
        legend_labels,
        title="Categories",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.3, 1),
        fontsize=8
    )

    # center text
    ax.text(0, 0, f"{custom_outer_label or outer_col}\n&\n{custom_inner_label or inner_col}",
            ha='center', va='center', fontsize=12)
    # chart display
    st.pyplot(fig)
    # download
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="Download as PNG",
        data=buf,
        file_name="nested_pie_chart.png",
        mime="image/png"
    )


# HISTOGRAM
def create_histogram(
    df,
    x_column,
    hue_column=None,
    filter_column=None,
    filter_value=None,
    custom_x_label=None,
    custom_y_label=None
):
    # filtering
    if filter_column and filter_value and filter_value != 'None':
        df = df[df[filter_column] == filter_value]

    # set labels
    prettify = lambda s: s.replace("_", " ").capitalize()
    xlabel = custom_x_label if custom_x_label else prettify(x_column)
    ylabel = custom_y_label if custom_y_label else "Frequency"

    filter_text = ""
    if filter_column and filter_value and filter_value != 'None':
        filter_text = f" – for '{filter_value}' in {prettify(filter_column)}"

    title_label = f"Histogram of {xlabel}{filter_text}"

    # plotting
    fig, ax = plt.subplots()
    sns.histplot(data=df, x=x_column, hue=(hue_column if hue_column != 'None' else None), kde=True, bins='auto', ax=ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("")  # Title moved to footer

    # footer description
    fig.text(0.5, -0.08, title_label, ha='center', fontsize=10, color='gray')

    # chart display
    st.pyplot(fig)
    # download
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="Download as PNG",
        data=buf,
        file_name="histogram.png",
        mime="image/png"
    )

# SCATTER PLOT
def create_scatter_plot(
    df,
    x_column,
    y_column,
    hue_column=None,
    filter_column=None,
    filter_value=None,
    custom_x_label=None,
    custom_y_label=None
):
    # filtering
    if filter_column and filter_value and filter_value != 'None':
        df = df[df[filter_column] == filter_value]

    # set labels
    prettify = lambda s: s.replace("_", " ").capitalize()
    xlabel = custom_x_label if custom_x_label else prettify(x_column)
    ylabel = custom_y_label if custom_y_label else prettify(y_column)

    filter_text = ""
    if filter_column and filter_value and filter_value != 'None':
        filter_text = f" – for '{filter_value}' in {prettify(filter_column)}"

    hue_text = f" colored by {prettify(hue_column)}" if hue_column and hue_column != 'None' else ""
    title = f"Scatter Plot: {ylabel} vs {xlabel}{hue_text}{filter_text}"

    # plotting
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x_column, y=y_column, hue=hue_column if hue_column != 'None' else None, ax=ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("")  # Remove top title

    # footer title
    fig.text(0.5, -0.08, title, ha='center', fontsize=10, color='gray')

    # chart display
    st.pyplot(fig)
    # download
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="Download as PNG",
        data=buf,
        file_name="simple_bar.png",
        mime="image/png"
    )


def load_lottieurl(url:str):
    r = requests.get(url)
    if r.status_code!=200:
        return None
    return r.json()

# pages
# HOME PAGE
def show_home_page():

    st.markdown("""
        <style>
        @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css');
        .centered {
            text-align: center;
            font-family: 'Arial', sans-serif;
        }
        .icon {
            color: #4CAF50;  /* Example color */
        }
        </style>
        <div class="centered">
            <h1><i class="fas fa-bar-chart-o icon"></i> Welcome to Plotter!</h1>
            <p><em>Plotter is a web app that allows you create cool graphs based on your data in few steps<em></p>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("How to create plot?")
        st.subheader("Step 1. Upload your file(s)")
        st.subheader("Step 2. Preview your data")
        st.subheader("Step 3. Analyze descriptive statistics")
        st.subheader("Step 4. Customize your plot")
        st.subheader("Step 5. Create a plot!")


    with col2:
        lottie_graph = load_lottieurl("https://lottie.host/66722155-7642-4d7d-aa01-4834e7ec7ba8/0uhwqsYGce.json")
        if lottie_graph:
            st_lottie(lottie_graph, height=400, width=800)

# Embed FontAwesome and Custom Styles
    st.markdown("""
<style>
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css');

.centered {
    text-align: center;
    font-family: 'Arial', sans-serif;
}

.icon {
    color: #4CAF50;  /* Example color */
}
</style>
""", unsafe_allow_html=True)


# identifying optional / required hue charts (grouping)
required_hue_charts = ['Multiple Bar Chart', 'Stacked Bar Chart', 'Nested Pie Chart']
optional_hue_charts = ['Line Chart', 'Scatter Plot', 'Histogram']

# PROJECTS PAGE
def show_projects_page():

    st.markdown("""
    <style>
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css');
    .centered {
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    .icon {
        color: #4CAF50;  /* Example color */
    }
    </style>
    <div class="centered">
        <h1><i class="fas fa-bar-chart-o icon"></i> Let's create a plot!</h1>
        <p><em>Move step-by-step from 1 to 5 and enjoy.<em></p>
    </div>
    """, unsafe_allow_html=True)

    df = None

  # file uploader
    st.write('### Step 1. Upload your file.')
    uploaded_files = st.file_uploader(
        "Choose a file(s) from your computer or drag&drop…", accept_multiple_files=True, type=['xlsx', 'xls', 'csv']
    )

    dfs = {}
    file_names = ['Not selected']

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name

        # csv and excel
        if file_name.lower().endswith('.csv'):
            df_tmp = pd.read_csv(uploaded_file, thousands=',', decimal='.')
        else:
            df_tmp = pd.read_excel(uploaded_file)

        # clean up all object-dtype columns
        for col in df_tmp.columns:
            if df_tmp[col].dtype == 'object':
                # strip out $,% and commas, whitespace
                cleaned = (
                    df_tmp[col]
                    .astype(str)
                    .str.replace(r'[\$,%]', '', regex=True)
                    .str.strip()
                )
                # try parsing to numeric (bad parses → NaN)
                converted = pd.to_numeric(cleaned, errors='coerce')
                # only overwrite if >90% of values successfully parsed
                if converted.notna().mean() > 0.9:
                    df_tmp[col] = converted

        dfs[file_name] = df_tmp
        file_names.append(file_name)

    selected_file = st.selectbox('Choose a file to work with...', options=file_names)
    if selected_file != 'Not selected':
        df = dfs[selected_file]

  # data preview
    st.write("### Step 2. Preview the dataset")
    if selected_file and selected_file != 'Not selected':
        with st.spinner('Wait a sec...'):
            time.sleep(2)  # simulate a delay for processing
            with st.expander("Preview your dataset.", expanded=False):
            # display the dataFrame of the selected file
                st.dataframe(dfs[selected_file], height=300, width=700)
    else:
        # placeholder text if no file is selected
        st.info("Preview of selected file will be displayed here. Choose a file first.")


    # descriptive statistics display
    st.write('### Step 3. Descriptive statistics')
    if selected_file and selected_file != 'Not selected':
        with st.spinner('Wait a sec...'):
            time.sleep(2)
            with st.expander("View descriptive stats.", expanded=False):
                st.write("**Shape:**", df.shape)
                st.write("**Data types:**")
                st.dataframe(pd.DataFrame(df.dtypes, columns=["dtype"]).T)

                # numeric summary
                st.write("**Numeric summary:**")
                st.dataframe(df.describe().T)

                # categorical summary
                cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
                if len(cat_cols):
                    st.write("**Categorical summary:**")
                    st.dataframe(df[cat_cols].describe().T)

                # missing values
                st.write("**Missing values per column:**")
                st.dataframe(df.isnull().sum().to_frame("n_missing"))
    else:
        # placeholder text if no file is selected
        st.info("Descriptive statistics of your file will be here. Choose a file first.")


    col4_header, col5_header = st.columns([1, 1])
    with col4_header:
        st.write("### Step 4. Customize your plot")
    with col5_header:
        st.write("### Step 5. Your chart")


    col_custom, col_showplot = st.columns([1, 1])
    if selected_file and selected_file != 'Not selected':
      with col_custom:
          plot_type = st.selectbox(
              "Select Plot Type:",
              [
                  'Not selected', 'Simple Bar Chart', 'Horizontal Bar Chart',
                  'Multiple Bar Chart', 'Stacked Bar Chart',
                  'Line Chart', 'Pie Chart', 'Nested Pie Chart',
                  'Histogram', 'Scatter Plot'
              ]
          )

          # sorting only for simple/horizontal bars
          sortable_chart_types = ['Simple Bar Chart', 'Horizontal Bar Chart']
          sort_order = None
          if plot_type in sortable_chart_types:
              sort_order = st.selectbox("Sort Y-axis values:", ["None", "Ascending", "Descending"])

          # build column profile
          numeric_cols = df.select_dtypes(include='number').columns.tolist()
          categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
          datetime_cols = df.select_dtypes(include='datetime64[ns]').columns.tolist()
          profile = {
              'numeric': numeric_cols,
              'categorical': categorical_cols,
              'datetime': datetime_cols
          }
          col_opts = get_eligible_columns_from_profile(profile, df, plot_type)

          # placeholders for every selector
          x_column = y_column = hue_column = None
          outer_col = inner_col = None

          # nested pie chart ui
          if plot_type == 'Nested Pie Chart':
              outer_col = st.selectbox(
                  "Select column for outer ring:",
                  ['Not selected'] + col_opts["x"]
              )
              inner_choices = [c for c in col_opts["hue"] if c != outer_col]
              inner_col = st.selectbox(
                  "Select column for inner ring:",
                  ['Not selected'] + inner_choices
              )
              # custom labels
              custom_outer_label = st.text_input("Custom label for outer ring (Optional):", "",
                                                 key="nested_outer_label")
              custom_inner_label = st.text_input("Custom label for inner ring (Optional):", "",
                                                 key="nested_inner_label")

              # optional filter
              filter_column = st.selectbox("Filter by Column (Optional):", ['None'] + categorical_cols)
              filter_value = None
              if filter_column != 'None':
                  vals = df[filter_column].dropna().unique().tolist()
                  filter_value = st.selectbox(f"Select value in {filter_column}:", ['None'] + sorted(map(str, vals)))

              # requirement check
              missing = []
              if outer_col == 'Not selected':   missing.append("outer ring")
              if inner_col == 'Not selected':   missing.append("inner ring")

              if missing:
                  st.warning("Please pick " + " and ".join(missing) + " values to create a Chart.")
                  button_clicked = False
              else:
                  button_clicked = st.button("Create Chart")

              # map into generic vars for downstream
              x_column = outer_col
              y_column = inner_col
              custom_x_label = custom_outer_label
              custom_y_label = custom_inner_label
              hue_column = None
              y_agg_func = None

          # all charts except nested pie chart ui
          elif plot_type != 'Not selected':
              x_column = st.selectbox("Select column for X-axis:", ['Not selected'] + col_opts["x"])

              if col_opts["y"]:
                  y_column = st.selectbox("Select column for Y-axis:", ['Not selected'] + col_opts["y"])


              if col_opts["hue"]:
                  # decide whether hue is required for this plot type
                  if plot_type in required_hue_charts:
                      hue_label = "Group data by...(Required):"
                      hue_options = col_opts["hue"]  # no 'None'
                  else:
                      hue_label = "Group data by...(Optional):"
                      hue_options = ['None'] + col_opts["hue"]

                  hue_column = st.selectbox(
                      hue_label,
                      hue_options,
                      key='hue_column'
                  )
              else:
                  hue_column = None

              # aggregation
              if plot_type in ['Simple Bar Chart', 'Horizontal Bar Chart'] and y_column and y_column != 'Not selected':
                  # default to 'mean'
                  y_agg_func = st.selectbox(
                      "Select aggregation function for Y-axis:",
                      ["mean", "sum", "median", "count"], index=0
                  )
              else:
                  y_agg_func = "mean"

              # optional filter
              filter_column = st.selectbox("Filter by Column (Optional):", ['None'] + categorical_cols,
                                           key="filter_col")
              filter_value = None
              if filter_column != 'None':
                  uvals = df[filter_column].dropna().unique().tolist()
                  filter_value = st.selectbox(f"Select value in {filter_column}:", ['None'] + sorted(map(str, uvals)),
                                              key="filter_val")

              # custom axis labels
              custom_x_label = st.text_input("Custom label for X-axis (Optional):", "")
              custom_y_label = st.text_input("Custom label for Y-axis (Optional):", "")

              # requirement check
              missing = []
              if x_column == 'Not selected': missing.append("X-axis")
              # these chart-types need a Y
              need_y = ['Simple Bar Chart', 'Horizontal Bar Chart',
                        'Multiple Bar Chart', 'Stacked Bar Chart',
                        'Line Chart', 'Scatter Plot']
              if plot_type in need_y and (not y_column or y_column == 'Not selected'):
                  missing.append("Y-axis")
              # some need a hue
              need_hue = ['Multiple Bar Chart', 'Stacked Bar Chart']
              if plot_type in need_hue and (not hue_column or hue_column == 'None'):
                  missing.append("grouping")

              if missing:
                  st.warning("Please select " + " and ".join(missing) + " to create a Chart.")
                  button_clicked = False
              else:
                  button_clicked = st.button("Create Plot")

          else:
              # plot_type == 'Not selected'
              st.info("Pick a plot type first.")
              button_clicked = False

      # CHART DISPLAY
      with col_showplot:
        chart_placeholder = st.empty()

        if not button_clicked:
          st.info("Your chart will appear here once you complete Step 4.")
        else:
          with st.spinner('Processing data...'):
            time.sleep(2)
            chart_placeholder.empty()

            if x_column != 'Not selected' and plot_type != 'Not selected':

                if plot_type == 'Simple Bar Chart':
                    create_simple_bar_chart(
                        df=df,
                        x_column=x_column,
                        y_column=y_column,
                        y_agg_func=y_agg_func,
                        aggregate_y=True,
                        filter_column=filter_column,
                        filter_value=filter_value,
                        sort_order=sort_order,
                        show_values=True
                    )
                elif plot_type == 'Horizontal Bar Chart':
                    create_horizontal_bar_chart(
                        df=df,
                        x_column=x_column,
                        y_column=y_column,
                        y_agg_func=y_agg_func,
                        aggregate_y=True,
                        filter_column=filter_column,
                        filter_value=filter_value,
                        sort_order=sort_order,
                        show_values=True,
                        custom_x_label=custom_x_label,
                        custom_y_label=custom_y_label
                    )
                elif plot_type == 'Multiple Bar Chart':
                    create_multiple_bar_chart(
                        df=df,
                        x_column=x_column,
                        y_column=y_column,
                        hue_column=hue_column,
                        y_agg_func=y_agg_func,
                        filter_column=filter_column,
                        filter_value=filter_value,
                        show_values=True,
                        custom_x_label=custom_x_label,
                        custom_y_label=custom_y_label
                    )
                elif plot_type == 'Stacked Bar Chart':
                    create_stacked_bar_chart(
                        df=df,
                        x_column=x_column,
                        y_column=y_column,
                        hue_column=hue_column,
                        y_agg_func=y_agg_func,
                        filter_column=filter_column,
                        filter_value=filter_value,
                        show_values=True,
                        custom_x_label=custom_x_label,
                        custom_y_label=custom_y_label
                    )
                elif plot_type == 'Line Chart':
                    create_line_chart(
                        df=df,
                        x_column=x_column,
                        y_column=y_column,
                        y_agg_func=y_agg_func,
                        aggregate_y=True,
                        hue_column=hue_column,
                        filter_column=filter_column,
                        filter_value=filter_value,
                        show_values=True,
                        custom_x_label=custom_x_label,
                        custom_y_label=custom_y_label
                    )
                elif plot_type == 'Pie Chart':
                    create_pie_chart(
                        df=df,
                        x_column=x_column,
                        filter_column=filter_column,
                        filter_value=filter_value,
                        custom_x_label=custom_x_label
                    )

                elif plot_type == 'Nested Pie Chart':
                    create_nested_pie_chart(
                        df=df,
                        outer_col=outer_col,
                        inner_col=inner_col,
                        filter_column=filter_column,
                        filter_value=filter_value,
                        custom_outer_label=custom_x_label,
                        custom_inner_label=custom_y_label
                    )

                elif plot_type == 'Histogram':
                    create_histogram(
                        df=df,
                        x_column=x_column,
                        hue_column=hue_column,
                        filter_column=filter_column,
                        filter_value=filter_value,
                        custom_x_label=custom_x_label,
                        custom_y_label=custom_y_label
                    )
                elif plot_type == 'Scatter Plot':
                    create_scatter_plot(
                        df=df,
                        x_column=x_column,
                        y_column=y_column,
                        hue_column=hue_column,
                        filter_column=filter_column,
                        filter_value=filter_value,
                        custom_x_label=custom_x_label,
                        custom_y_label=custom_y_label
                    )
                else:
                    st.error("Please choose columns and a plot type to generate a plot.")

    else:
        # no file selected - finish Step 3
        with col_custom:
            st.info("Complete previous steps to proceed to chart customization.")
        with col_showplot:
            st.info("Your chart will appear here once you complete Step 4.")



selected = option_menu(menu_title=None, options=["Home", "Projects"], icons=['house', 'bar-chart'], default_index=0, orientation="horizontal")
if selected == "Home":
    show_home_page()
if selected == "Projects":
    show_projects_page()

# footer of the website
st.markdown(
    """
    <style>
      .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px 0;
        background-color: #f0f2f6;
        color: #888888;
        font-size: 0.9em;
      }
      /* make sure the main content isn't hidden behind the footer */
      .element-container {
        padding-bottom: 50px;
      }
    </style>
    <div class="footer">
       Plotter App © 2025. Developed by Assem Agabekova as a thesis project at KIMEP University.
    </div>
    """,
    unsafe_allow_html=True,
)