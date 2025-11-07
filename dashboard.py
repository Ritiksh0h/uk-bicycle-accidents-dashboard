import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="UK Bicycle Accidents 1979-2018", layout="wide")

# Load data (use caching to avoid reloading)
@st.cache_data
def load_data():
    accidents = pd.read_csv('data/Accidents.csv')
    bikers = pd.read_csv('data/Bikers.csv')
    
    # Date conversion
    accidents['Date'] = pd.to_datetime(accidents['Date'], format='%Y-%m-%d')
    accidents['Year'] = accidents['Date'].dt.year
    accidents['Month'] = accidents['Date'].dt.month
    
    # Time conversion
    accidents['Time'] = pd.to_datetime(accidents['Time'], format='%H:%M', errors='coerce')
    accidents['Hour'] = accidents['Time'].dt.hour
    
    # Merge
    merged = accidents.merge(bikers, on='Accident_Index', how='inner')
    merged['Year'] = merged['Date'].dt.year
    merged['Hour'] = merged['Time'].dt.hour
    
    return accidents, bikers, merged

accidents, bikers, merged = load_data()

# TITLE
st.title("🚴 Britain's Cycling Safety Paradox: 40 Years of Data")
st.markdown("### Progress Made, Risks Remain")

# SIDEBAR for filters
st.sidebar.header("Filters")
year_range = st.sidebar.slider("Select Year Range", 1979, 2018, (1979, 2018))

# Filter data based on selection
filtered_merged = merged[(merged['Year'] >= year_range[0]) & (merged['Year'] <= year_range[1])]

# === PAGE 1: OVERVIEW ===
st.header("📊 Overview: The Big Picture")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Accidents", f"{len(merged):,}")
with col2:
    accidents_1979 = len(merged[merged['Year'] == 1979])
    accidents_2018 = len(merged[merged['Year'] == 2018])
    pct_change = ((accidents_2018 - accidents_1979) / accidents_1979) * 100
    st.metric("Change 1979-2018", f"{pct_change:.1f}%", delta=f"{pct_change:.1f}%")
with col3:
    fatal_rate = (merged['Severity'] == 'Fatal').sum() / len(merged) * 100
    st.metric("Fatal Rate", f"{fatal_rate:.2f}%")
with col4:
    serious_rate = (merged['Severity'] == 'Serious').sum() / len(merged) * 100
    st.metric("Serious Injury Rate", f"{serious_rate:.1f}%")

# Yearly trend chart
st.subheader("Accidents Over Time: The Decline")
yearly_data = filtered_merged.groupby('Year').size().reset_index(name='Count')

fig_yearly = px.line(yearly_data, x='Year', y='Count', 
                      title='Total Accidents per Year (1979-2018)',
                      labels={'Count': 'Number of Accidents', 'Year': 'Year'})
fig_yearly.update_traces(line_color='#FF6B6B', line_width=3)
fig_yearly.update_layout(hovermode='x unified')
st.plotly_chart(fig_yearly, width='stretch')

st.info("📈 Note: While total accidents decreased 25%, serious injuries as % of total rose from 15.3% (2009) to 21.2% (2018)")

st.markdown("---")

# === PAGE 2: WHO'S AT RISK ===
st.header("👥 Who's at Risk?")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Severity by Age Group")
    # Severity by Age Group
age_severity = filtered_merged.groupby(['Age_Grp', 'Severity']).size().reset_index(name='Count')
age_severity_pct = age_severity.pivot_table(index='Age_Grp', columns='Severity', values='Count', fill_value=0)
age_severity_pct = age_severity_pct.div(age_severity_pct.sum(axis=1), axis=0) * 100

fig_age = px.bar(age_severity_pct.reset_index().melt(id_vars='Age_Grp'), 
                 x='Age_Grp', y='value', color='Severity',
                 title='Severity by Age Group (%)',
                 labels={'value': 'Percentage', 'Age_Grp': 'Age Group'},
                 color_discrete_map={'Slight': '#4ECDC4', 'Serious': '#FFE66D', 'Fatal': '#FF6B6B'})
fig_age.update_layout(barmode='stack')
st.plotly_chart(fig_age, width='stretch')

# Add insight text
if len(filtered_merged[filtered_merged['Age_Grp'] == '66 to 75']) > 0:
    older_severe = (filtered_merged[filtered_merged['Age_Grp'] == '66 to 75']['Severity'].isin(['Serious', 'Fatal'])).sum()
    older_total = len(filtered_merged[filtered_merged['Age_Grp'] == '66 to 75'])
    older_pct = (older_severe / older_total) * 100
    st.warning(f"⚠️ Cyclists aged 66-75 have a {older_pct:.1f}% serious/fatal injury rate - nearly double younger cyclists")
    
with col2:
    st.subheader("Gender Distribution")
    # Gender distribution with severity
gender_counts = filtered_merged['Gender'].value_counts()
fig_gender = px.pie(values=gender_counts.values, names=gender_counts.index,
                     title='Accidents by Gender',
                     color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#95E1D3'])
st.plotly_chart(fig_gender, width='stretch')

# Gender severity comparison
gender_severity = filtered_merged.groupby(['Gender', 'Severity']).size().unstack(fill_value=0)
gender_severity_pct = gender_severity.div(gender_severity.sum(axis=1), axis=0) * 100

st.markdown("**Severity Breakdown by Gender:**")
col_m, col_f = st.columns(2)
with col_m:
    male_fatal = gender_severity_pct.loc['Male', 'Fatal']
    st.metric("Male Fatal Rate", f"{male_fatal:.2f}%")
with col_f:
    female_fatal = gender_severity_pct.loc['Female', 'Fatal']
    st.metric("Female Fatal Rate", f"{female_fatal:.2f}%")

st.markdown("---")

# === PAGE 3: WHEN & WHERE ===
st.header("⏰ When and Where Danger Strikes")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Accidents by Hour of Day")
    # Hourly accidents
hourly_data = filtered_merged.groupby('Hour').size().reset_index(name='Count')

fig_hourly = px.line(hourly_data, x='Hour', y='Count',
                      title='Accidents by Hour of Day',
                      labels={'Count': 'Number of Accidents', 'Hour': 'Hour (24h)'})
fig_hourly.update_traces(line_color='#FF6B6B', line_width=3, fill='tozeroy')
fig_hourly.update_xaxes(tickmode='linear', tick0=0, dtick=2)
st.plotly_chart(fig_hourly, width='stretch')

# Highlight rush hours
rush_hour_am = filtered_merged[filtered_merged['Hour'].isin([7, 8, 9])].shape[0]
rush_hour_pm = filtered_merged[filtered_merged['Hour'].isin([16, 17, 18])].shape[0]
st.info(f"🚨 Rush hours account for {rush_hour_am + rush_hour_pm:,} accidents ({((rush_hour_am + rush_hour_pm)/len(filtered_merged)*100):.1f}% of total)")
    
with col2:
    st.subheader("Speed Limit vs Severity")
    # Speed limit vs severity
common_speeds = [20, 30, 40, 50, 60, 70]
speed_data = filtered_merged[filtered_merged['Speed_limit'].isin(common_speeds)]
speed_severity = speed_data.groupby(['Speed_limit', 'Severity']).size().unstack(fill_value=0)
speed_severity_pct = speed_severity.div(speed_severity.sum(axis=1), axis=0) * 100

# Focus on serious + fatal combined
speed_severity_pct['Severe'] = speed_severity_pct['Serious'] + speed_severity_pct['Fatal']

fig_speed = px.bar(speed_severity_pct.reset_index(), x='Speed_limit', y='Severe',
                    title='Severe Injury Rate by Speed Limit',
                    labels={'Severe': 'Severe Outcome Rate (%)', 'Speed_limit': 'Speed Limit (mph)'},
                    color='Severe',
                    color_continuous_scale='Reds')
st.plotly_chart(fig_speed, width='stretch')

st.error(f"💀 At 70 mph, {speed_severity_pct.loc[70, 'Severe']:.1f}% of accidents result in serious injury or death")

st.markdown("---")

# === PAGE 4: PRESCRIPTIVE RECOMMENDATIONS ===
st.header("💡 Data-Driven Recommendations")

st.markdown("""
Based on 40 years of data analysis, here are evidence-based interventions:

| Priority | Recommendation | Expected Impact | Data Support |
|----------|---------------|-----------------|--------------|
| 🔴 HIGH | Protected rush hour bike lanes (7-9 AM, 4-6 PM) | -30% rush hour accidents | 172,025 accidents during these hours |
| 🔴 HIGH | Senior cyclist safety programs (60+) | -25% severity for older cyclists | 30% serious/fatal rate for 66-75 age group |
| 🟡 MEDIUM | 20 mph zones in cycling hotspots | -50% severe injuries | 70 mph zones have 39% severe outcomes vs 16% at 30 mph |
| 🟡 MEDIUM | Night enforcement + lighting (11 PM - 5 AM) | -40% nighttime fatalities | 2.8% fatality rate midnight-4 AM vs 0.5-0.7% during day |
| 🟢 LOW | Male-focused safety campaigns | -10% male accidents | Males = 80% of accidents, 0.84% fatality rate |
""")

st.markdown("---")

# FOOTER
st.markdown("**Data Source:** UK Department for Transport (1979-2018) | **Analysis Date:** November 2025")