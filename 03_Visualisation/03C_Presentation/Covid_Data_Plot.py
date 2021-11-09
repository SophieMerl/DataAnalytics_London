####### Load packages
import pandas as pd
import numpy as np
import io
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

####### Set path
url_cov ="https://raw.githubusercontent.com/SophieMerl/DataAnalytics_London/master/data/London_Covid_301021.csv"
download_cov = requests.get(url_cov).content
df_cov = pd.read_csv(io.StringIO(download_cov.decode('utf-8')))

####### Plotting
##Cases
df_cov['datetime'] = pd.to_datetime(df_cov['Date'])

fig, ax = plt.subplots()
ax.plot('datetime', 'newCasesBySpecimenDate', data=df_cov)

# Major ticks every 3 months.
fmt_half_year = mdates.MonthLocator(interval=3)
ax.xaxis.set_major_locator(fmt_half_year)

# Minor ticks every month.
fmt_month = mdates.MonthLocator()
ax.xaxis.set_minor_locator(fmt_month)

# Text in the x axis will be displayed in 'YYYY-mm' format.
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.format_xdata = mdates.DateFormatter('%Y-%m')
ax.grid(True)

# Rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them.
fig.autofmt_xdate()

plt.xlabel('Date')
plt.ylabel('New cases')
plt.show()

##Vaccines
df_cov['FirstVaccine_perc'] = df_cov['cumPeopleVaccinatedFirstDoseByVaccinationDate']/8982000*100
df_cov['SecVaccine_perc'] = df_cov['cumPeopleVaccinatedSecondDoseByVaccinationDate']/8982000*100

fig, ax = plt.subplots()
ax.plot('datetime', 'FirstVaccine_perc', data=df_cov, color='lightblue', label = "First vaccination")
ax.plot('datetime', 'SecVaccine_perc', data=df_cov, color= 'blue', label = "Second vaccination")

# Major ticks every 3 months.
fmt_half_year = mdates.MonthLocator(interval=2)
ax.xaxis.set_major_locator(fmt_half_year)

# Minor ticks every month.
fmt_month = mdates.MonthLocator()
ax.xaxis.set_minor_locator(fmt_month)

# Text in the x axis will be displayed in 'YYYY-mm' format.
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.format_xdata = mdates.DateFormatter('%Y-%m')
ax.grid(True)

# Rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them.
fig.autofmt_xdate()

plt.legend()
plt.xlabel('Date')
plt.ylabel('Population London (%)')

plt.show()

##Testing
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax2.plot('datetime', 'uniqueCasePositivityBySpecimenDateRollingSum', data=df_cov[:621], color='lightgreen', label = "Test positivity")
ax.plot('datetime', 'uniquePeopleTestedBySpecimenDateRollingSum', data=df_cov[:621], color= 'darkgreen', label = "People tested")

# Major ticks every 3 months.
fmt_half_year = mdates.MonthLocator(interval=2)
ax.xaxis.set_major_locator(fmt_half_year)

# Minor ticks every month.
fmt_month = mdates.MonthLocator()
ax.xaxis.set_minor_locator(fmt_month)

# Text in the x axis will be displayed in 'YYYY-mm' format.
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# Format the coords message box, i.e. the numbers displayed as the cursor moves
# across the axes within the interactive GUI.
ax.format_xdata = mdates.DateFormatter('%Y-%m')
ax.grid(True)

ax.set_xlabel('Date')
ax.set_ylabel('Tests', color='darkgreen')
ax2.set_ylabel('Test positivity (%)', color='lightgreen')

# Rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them.
fig.autofmt_xdate()

plt.show()

##Deaths
df_cov['datetime'] = pd.to_datetime(df_cov['Date'])

fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.plot('datetime', 'newDeaths28DaysByDeathDate', data=df_cov, color='black', label = "New deaths")
ax2.plot('datetime', 'cumDeaths28DaysByDeathDate', data=df_cov, color='grey', label = "Deaths cumulated")

# Major ticks every 3 months.
fmt_half_year = mdates.MonthLocator(interval=3)
ax.xaxis.set_major_locator(fmt_half_year)

# Minor ticks every month.
fmt_month = mdates.MonthLocator()
ax.xaxis.set_minor_locator(fmt_month)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

ax.format_xdata = mdates.DateFormatter('%Y-%m')
ax.grid(True)

# Rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them.
ax.set_xlabel('Date')
ax.set_ylabel('New deaths', color='black')
ax2.set_ylabel('Deaths cumulated', color='grey')
fig.autofmt_xdate()
plt.show()

##Healthcare
fig, ax = plt.subplots()
ax.plot('datetime', 'newAdmissions', data=df_cov, color='pink', label = "Admissions")
ax.plot('datetime', 'hospitalCases', data=df_cov, color= 'purple', label = "Hospital cases")
ax.plot('datetime', 'covidOccupiedMVBeds', data=df_cov, color= 'red', label = "Occupied ventilation beds")

# Major ticks every 3 months.
fmt_half_year = mdates.MonthLocator(interval=2)
ax.xaxis.set_major_locator(fmt_half_year)

# Minor ticks every month.
fmt_month = mdates.MonthLocator()
ax.xaxis.set_minor_locator(fmt_month)

# Text in the x axis will be displayed in 'YYYY-mm' format.
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# Format the coords message box, i.e. the numbers displayed as the cursor moves
# across the axes within the interactive GUI.
ax.format_xdata = mdates.DateFormatter('%Y-%m')
#ax.format_ydata = lambda x: f'${x:.2f}'  # Format the price.
ax.grid(True)

# Rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them.
fig.autofmt_xdate()
plt.legend()
plt.xlabel('Date')
plt.ylabel('Patients')
plt.show()

