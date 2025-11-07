import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print("="*80)
print("BICYCLE ACCIDENTS IN GREAT BRITAIN (1979-2018) - EDA")
print("="*80)

# Load data
print("\n[1] LOADING DATA...")
accidents = pd.read_csv('data/Accidents.csv')
bikers = pd.read_csv('data/Bikers.csv')

print(f"Accidents shape: {accidents.shape}")
print(f"Bikers shape: {bikers.shape}")

# Basic info
print("\n[2] DATA STRUCTURE")
print("\nAccidents columns:")
print(accidents.dtypes)
print("\nBikers columns:")
print(bikers.dtypes)

# Missing values
print("\n[3] MISSING VALUES CHECK")
print("\nAccidents missing:")
print(accidents.isnull().sum())
print("\nBikers missing:")
print(bikers.isnull().sum())

# Convert Date column to datetime - FIXED FORMAT
print("\n[4] DATE CONVERSION")
accidents['Date'] = pd.to_datetime(accidents['Date'], format='%Y-%m-%d', errors='coerce')
accidents['Year'] = accidents['Date'].dt.year
accidents['Month'] = accidents['Date'].dt.month
accidents['DayOfWeek'] = accidents['Date'].dt.day_name()

print("Date range:", accidents['Date'].min(), "to", accidents['Date'].max())

# Time trends
print("\n[5] ACCIDENTS OVER TIME")
yearly_accidents = accidents.groupby('Year').size().reset_index(name='Count')
print("\nAccidents by Year:")
print(yearly_accidents)

print("\n% Change from 1979 to 2018:")
first_year = yearly_accidents[yearly_accidents['Year'] == 1979]['Count'].values[0]
last_year = yearly_accidents[yearly_accidents['Year'] == 2018]['Count'].values[0]
pct_change = ((last_year - first_year) / first_year) * 100
print(f"{pct_change:.1f}%")

# Monthly patterns
print("\n[6] MONTHLY PATTERNS")
monthly = accidents.groupby('Month').size().reset_index(name='Count')
monthly['Month_Name'] = monthly['Month'].apply(lambda x: datetime(2000, x, 1).strftime('%B'))
print(monthly[['Month_Name', 'Count']])

# Day of week
print("\n[7] DAY OF WEEK DISTRIBUTION")
day_counts = accidents['Day'].value_counts()
print(day_counts)

# Time of day analysis
print("\n[8] TIME OF DAY ANALYSIS")
accidents['Time'] = pd.to_datetime(accidents['Time'], format='%H:%M', errors='coerce')
accidents['Hour'] = accidents['Time'].dt.hour
hourly = accidents.groupby('Hour').size().reset_index(name='Count')
print("\nAccidents by Hour:")
print(hourly)

# Speed limit analysis
print("\n[9] SPEED LIMIT ANALYSIS")
speed_counts = accidents['Speed_limit'].value_counts().sort_index()
print(speed_counts)

# Road conditions
print("\n[10] ROAD CONDITIONS")
print(accidents['Road_conditions'].value_counts())
print("\nWeather conditions:")
print(accidents['Weather_conditions'].value_counts())

# Road type
print("\n[11] ROAD TYPE")
print(accidents['Road_type'].value_counts())

# Light conditions
print("\n[12] LIGHT CONDITIONS")
print(accidents['Light_conditions'].value_counts())

# Casualties analysis
print("\n[13] CASUALTIES & SEVERITY")
print("\nNumber of casualties distribution:")
print(accidents['Number_of_Casualties'].describe())
print("\nCasualties breakdown:")
print(accidents['Number_of_Casualties'].value_counts().head(10))

# Join with bikers data
print("\n[14] MERGING BIKER DEMOGRAPHICS")
merged = accidents.merge(bikers, on='Accident_Index', how='inner')
print(f"Merged dataset shape: {merged.shape}")

# Gender analysis
print("\n[15] GENDER ANALYSIS")
gender_counts = merged['Gender'].value_counts()
print(gender_counts)
print("\nGender percentages:")
print(merged['Gender'].value_counts(normalize=True) * 100)

# Severity analysis
print("\n[16] SEVERITY ANALYSIS")
severity_counts = merged['Severity'].value_counts()
print(severity_counts)
print("\nSeverity percentages:")
print(merged['Severity'].value_counts(normalize=True) * 100)

# Age group analysis
print("\n[17] AGE GROUP ANALYSIS")
age_counts = merged['Age_Grp'].value_counts().sort_index()
print(age_counts)

# Cross-tabulations for insights
print("\n[18] KEY CROSS-TABULATIONS")

print("\nSeverity by Gender:")
print(pd.crosstab(merged['Gender'], merged['Severity'], normalize='index') * 100)

print("\nSeverity by Age Group:")
severity_age = pd.crosstab(merged['Age_Grp'], merged['Severity'], normalize='index') * 100
print(severity_age)

print("\nAge Group by Gender:")
age_gender = pd.crosstab(merged['Age_Grp'], merged['Gender'], margins=True)
print(age_gender)

# Weather vs severity
print("\n[19] WEATHER CONDITIONS VS SEVERITY")
weather_severity = pd.crosstab(merged['Weather_conditions'], merged['Severity'], normalize='index') * 100
print(weather_severity.round(1))

# Road conditions vs severity
print("\n[20] ROAD CONDITIONS VS SEVERITY")
road_severity = pd.crosstab(merged['Road_conditions'], merged['Severity'], normalize='index') * 100
print(road_severity.round(1))

# Speed limit vs severity
print("\n[21] SPEED LIMIT VS SEVERITY")
merged_speed = merged[merged['Speed_limit'].isin([20, 30, 40, 50, 60, 70])]
speed_severity = pd.crosstab(merged_speed['Speed_limit'], merged_speed['Severity'], normalize='index') * 100
print(speed_severity.round(1))

print("\n[22] TIME OF DAY VS SEVERITY")
merged['Hour'] = merged['Time'].dt.hour
time_severity = merged.groupby('Hour')['Severity'].value_counts(normalize=True).unstack() * 100
print(time_severity.round(1))

# Trends over time
print("\n[23] SEVERITY TRENDS OVER TIME")
merged['Year'] = merged['Date'].dt.year
yearly_severity = merged.groupby(['Year', 'Severity']).size().unstack(fill_value=0)
yearly_severity_pct = yearly_severity.div(yearly_severity.sum(axis=1), axis=0) * 100
print(yearly_severity_pct.tail(10))

print("\n[24] GENDER TRENDS OVER TIME")
yearly_gender = merged.groupby(['Year', 'Gender']).size().unstack(fill_value=0)
print(yearly_gender.tail(10))

print("\n" + "="*80)
print("EDA COMPLETE - REVIEW OUTPUT ABOVE")
print("="*80)