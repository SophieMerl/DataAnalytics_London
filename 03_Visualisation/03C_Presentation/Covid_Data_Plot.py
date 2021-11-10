####### Load packages
import pandas as pd
import numpy as np
import io
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

url_cov ="https://raw.githubusercontent.com/SophieMerl/DataAnalytics_London/master/02_Preprocessing/London_Covid_301021.csv"
download_cov = requests.get(url_cov).content
df_cov = pd.read_csv(io.StringIO(download_cov.decode('utf-8')))

#######
#Cases
df_cov['datetime'] = pd.to_datetime(df_cov['Date'])

fig, ax = plt.subplots()
ax.plot('datetime', 'newCasesBySpecimenDate', data=df_cov, color = "mediumblue", linewidth=.7)

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
#spring
plt.axvspan("2020-03-20", "2020-06-20", color='lawngreen', alpha=0.2, lw=0)
plt.axvspan("2021-03-20", "2021-06-20", color='lawngreen', alpha=0.2, lw=0)
#summer
plt.axvspan("2020-06-21", "2020-09-21", color='crimson', alpha=0.2, lw=0)
plt.axvspan("2021-06-21", "2021-09-21", color='crimson', alpha=0.2, lw=0)
#autumn
plt.axvspan("2020-09-22", "2020-12-21", color='coral', alpha=0.2, lw=0)
plt.axvspan("2021-09-22", "2021-12-01", color='coral', alpha=0.2, lw=0)
#winter
plt.axvspan("2020-02-01", "2020-03-19", color='lightskyblue', alpha=0.2, lw=0)
plt.axvspan("2020-12-22", "2021-03-19", color='lightskyblue', alpha=0.2, lw=0)

plt.show()

#Vaccines
df_cov['FirstVaccine_perc'] = df_cov['cumPeopleVaccinatedFirstDoseByVaccinationDate']/8982000*100
df_cov['SecVaccine_perc'] = df_cov['cumPeopleVaccinatedSecondDoseByVaccinationDate']/8982000*100

fig, ax = plt.subplots()
ax.plot('datetime', 'FirstVaccine_perc', data=df_cov, color='steelblue', label = "First vaccination")
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
#spring
plt.axvspan("2021-03-20", "2021-06-20", color='lawngreen', alpha=0.2, lw=0)
#summer
plt.axvspan("2021-06-21", "2021-09-21", color='crimson', alpha=0.2, lw=0)
#autumn
plt.axvspan("2020-12-01", "2020-12-21", color='coral', alpha=0.2, lw=0)
plt.axvspan("2021-09-22", "2021-11-02", color='coral', alpha=0.2, lw=0)
#winter
plt.axvspan("2020-12-22", "2021-03-19", color='lightskyblue', alpha=0.2, lw=0)
plt.show()

#Testing
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax2.plot('datetime', 'uniqueCasePositivityBySpecimenDateRollingSum', data=df_cov[:621], color='mediumseagreen', label = "Test positivity")
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
#ax.format_ydata = lambda x: f'${x:.2f}'  # Format the price.
ax.grid(True)

ax.set_xlabel('Date')
ax.set_ylabel('Tests', color='darkgreen')
ax2.set_ylabel('Test positivity (%)', color='mediumseagreen')

# Rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them.
fig.autofmt_xdate()
#spring
plt.axvspan("2020-03-20", "2020-06-20", color='lawngreen', alpha=0.2, lw=0)
plt.axvspan("2021-03-20", "2021-06-20", color='lawngreen', alpha=0.2, lw=0)
#summer
plt.axvspan("2020-06-21", "2020-09-21", color='crimson', alpha=0.2, lw=0)
plt.axvspan("2021-06-21", "2021-09-21", color='crimson', alpha=0.2, lw=0)
#autumn
plt.axvspan("2020-09-22", "2020-12-21", color='coral', alpha=0.2, lw=0)
plt.axvspan("2021-09-22", "2021-11-01", color='coral', alpha=0.2, lw=0)
#winter
plt.axvspan("2019-12-22", "2020-03-19", color='lightskyblue', alpha=0.2, lw=0)
plt.axvspan("2020-12-22", "2021-03-19", color='lightskyblue', alpha=0.2, lw=0)
plt.show()

#######
#Deaths
df_cov['datetime'] = pd.to_datetime(df_cov['Date'])

fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.plot('datetime', 'newDeaths28DaysByDeathDate', data=df_cov, color='black', label = "New deaths", linewidth = .9)
ax2.plot('datetime', 'cumDeaths28DaysByDeathDate', data=df_cov, color='grey', label = "Deaths cumulated", linewidth = .9)

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
#spring
plt.axvspan("2020-03-20", "2020-06-20", color='lawngreen', alpha=0.2, lw=0)
plt.axvspan("2021-03-20", "2021-06-20", color='lawngreen', alpha=0.2, lw=0)
#summer
plt.axvspan("2020-06-21", "2020-09-21", color='crimson', alpha=0.2, lw=0)
plt.axvspan("2021-06-21", "2021-09-21", color='crimson', alpha=0.2, lw=0)
#autumn
plt.axvspan("2020-09-22", "2020-12-21", color='coral', alpha=0.2, lw=0)
plt.axvspan("2021-09-22", "2021-11-01", color='coral', alpha=0.2, lw=0)
#winter
plt.axvspan("2020-02-01", "2020-03-19", color='lightskyblue', alpha=0.2, lw=0)
plt.axvspan("2020-12-22", "2021-03-19", color='lightskyblue', alpha=0.2, lw=0)
plt.show()

#######
#Healthcare

fig, ax = plt.subplots()
ax.plot('datetime', 'newAdmissions', data=df_cov, color='palevioletred', label = "Admissions")
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
ax.grid(True)

# Rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them.
fig.autofmt_xdate()
plt.legend()
plt.xlabel('Date')
plt.ylabel('Patients')
#spring
plt.axvspan("2020-03-20", "2020-06-20", color='lawngreen', alpha=0.2, lw=0)
plt.axvspan("2021-03-20", "2021-06-20", color='lawngreen', alpha=0.2, lw=0)
#summer
plt.axvspan("2020-06-21", "2020-09-21", color='crimson', alpha=0.2, lw=0)
plt.axvspan("2021-06-21", "2021-09-21", color='crimson', alpha=0.2, lw=0)
#autumn
plt.axvspan("2020-09-22", "2020-12-21", color='coral', alpha=0.2, lw=0)
plt.axvspan("2021-09-22", "2021-11-01", color='coral', alpha=0.2, lw=0)
#winter
plt.axvspan("2020-02-01", "2020-03-19", color='lightskyblue', alpha=0.2, lw=0)
plt.axvspan("2020-12-22", "2021-03-19", color='lightskyblue', alpha=0.2, lw=0)
plt.show()
