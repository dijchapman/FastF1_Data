import fastf1
import fastf1.plotting
from fastf1.core import Laps
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib import cm
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from timple.timedelta import strftimedelta
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
import ffmpeg
pd.options.mode.chained_assignment = None  # default='warn'


def getTeamColors(session_laps):
    team_colors = list()
    for index, lap in session_laps.iterlaps():
        color = fastf1.plotting.team_color(lap['Team'])
        team_colors.append(color)
    return team_colors


def telemetryLinesAnimation(x_data, speed_data, throttle_data, brake_data, driver_color):
    # creating a blank window
    # for the animation
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    ax1.set_ylim(0, max(speed_data) * 1.1)
    ax1.set_xlim(0, max(x_data))
    ax2.set_ylim(0, max(throttle_data) * 1.1)
    ax2.set_xlim(0, max(x_data))
    ax3.set_ylim(0, max(brake_data) * 1.1)
    ax3.set_xlim(0, max(x_data))

    # intialize two line objects (one in each axes)
    line1, = ax1.plot([], [], lw=2, color=driver_color)
    line2, = ax2.plot([], [], lw=2, color='g')
    line3, = ax3.plot([], [], lw=2, color='r')
    lines = [line1, line2, line3]

    # initialize the data arrays
    xdata, y1data, y2data, y3data = [], [], [], []

    def animate(i):
        # update the data
        xdata.append(x_data[i])
        y1data.append(speed_data[i])
        y2data.append(throttle_data[i])
        y3data.append(brake_data[i])

        # update the data of both line objects
        lines[0].set_data(xdata, y1data)
        lines[1].set_data(xdata, y2data)
        lines[2].set_data(xdata, y3data)

        return lines

    # calling the animation function
    anim = FuncAnimation(fig, animate, frames=len(x_data), interval=20, repeat=False)
    plt.show()


def telemetryAnimation(tel_data, frame_speed, driver_color, plt_title):
    x_data = tel_data['Distance']
    speed_data = tel_data['Speed']
    throttle_data = tel_data['Throttle']
    brake_data = tel_data['Brake']
    time_data = tel_data['Time']
    x_map = np.array(tel_data['X'].values)
    y_map = np.array(tel_data['Y'].values)
    # creating a blank window
    # for the animation
    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(plt_title, fontsize=14)
    gs = GridSpec(nrows=2, ncols=2)

    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_ylim(0, max(speed_data) * 1.1)
    ax1.set_xlim(0, max(x_data))

    line, = ax1.plot([], [], lw=2, color=driver_color)
    xdata, ydata = [], []

    ax2 = fig.add_subplot(gs[1, 0])
    bars = ax2.barh(['Throttle', 'Brake'], [100, 0], color=['g', 'r'], alpha=0.75)
    ax2.invert_yaxis()

    # show circuit map
    ax3 = fig.add_subplot(gs[1, 1])
    points = np.array([x_map, y_map]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    cmap = cm.get_cmap('Paired')
    lc_comp = LineCollection(segments, norm=plt.Normalize(1, cmap.N + 1), cmap=cmap)
    lc_comp.set_linewidth(4)
    plt.gca().add_collection(lc_comp)
    plt.axis('equal')
    plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)
    ln, = ax3.plot([], [], 'go')

    def animate(i):
        if i in tel_data.index:
            ln.set_data(x_map[i], y_map[i])

            xdata.append(x_data[i])
            ydata.append(speed_data[i])
            line.set_data(xdata, ydata)

            bars[0].set_width(throttle_data[i])
            bars[1].set_width(brake_data[i] * 100)

            time_seconds = time_data[i].total_seconds()

    # calling the animation function
    # anim = FuncAnimation(fig, animate, frames=len(x_data), interval=frame_speed, repeat=False)
    anim = FuncAnimation(fig, animate, frames=len(x_data), interval=20, repeat=False)
    plt.show()

    # saves the animation to mp4
    # anim.save('animation.mp4', writer='ffmpeg', fps=30)


def trackMap(tel, driver_color):
    x = np.array(tel['X'].values)
    y = np.array(tel['Y'].values)

    fig = plt.figure()

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    cmap = cm.get_cmap('Paired')
    lc_comp = LineCollection(segments, norm=plt.Normalize(1, cmap.N + 1), cmap=cmap)
    lc_comp.set_linewidth(4)
    plt.gca().add_collection(lc_comp)
    plt.axis('equal')
    plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)
    ln, = plt.plot([], [], 'go')

    def animate(i):
        # update the data
        ln.set_data(x[i], y[i])
        return ln,

    # calling the animation function
    anim = FuncAnimation(fig, animate, frames=len(x), interval=20, repeat=False)

    plt.show()


def lapTelemetryData(session, driver):
    driver_lap = session.laps.pick_driver(driver).pick_fastest()
    driver_tel = driver_lap.get_telemetry()
    driver_color = fastf1.plotting.team_color(driver_lap['Team'])

    previous_time = 0
    for index, tel in driver_tel.iterrows():
        if previous_time == 0:
            time_delta = tel['Time']
        else:
            time_delta = tel['Time'] - previous_time

        driver_tel.loc[index, 'TimeDelta'] = time_delta
        previous_time = tel['Time']

    mean_time_delta = driver_tel['TimeDelta'].mean().total_seconds()
    # print("Average time delta between data points = ", driver_tel['TimeDelta'].mean().total_seconds())
    # print(driver_tel)
    # trackMap(driver_lap.get_telemetry(), driver_color)

    plt_title = f"Fastest Lap Telemetry Data \n {driver} in {session.event['EventName']} {session.event.year} Q"
    telemetryAnimation(driver_tel, mean_time_delta * 1000, driver_color, plt_title)


def driverTelemetryComparisons(session):
    drivers = pd.unique(session.laps['Driver'])

    drivers_data = {
        'Driver': drivers,
        'Team Color': ['white'] * len(drivers),
        'Braking Distance': [0] * len(drivers),
        'Avg Braking Distance': [0] * len(drivers),
        'Avg Braking Speed Reduction': [0] * len(drivers),
        'Start Braking Moments': [0] * len(drivers),
        'Accelerating Distance': [0] * len(drivers),
        'Full Throttle Distance': [0] * len(drivers)
    }

    drivers_df = pd.DataFrame(drivers_data)
    drivers_df['Start Braking Moments'] = drivers_df['Start Braking Moments'].astype('object')

    for index, drv in drivers_df.iterrows():
        fastest_lap = session.laps.pick_driver(drv['Driver']).pick_fastest()
        if not pd.isnull(fastest_lap['Driver']):
            lap_telemetry = fastest_lap.get_car_data().add_distance()
            team_color = fastf1.plotting.team_color(fastest_lap['Team'])
            if not pd.isnull(team_color):
                drivers_df.loc[drivers_df['Driver'] == fastest_lap['Driver'], 'Team Color'] = team_color
            braking_distance = 0.0
            braking_speed_start = 0.0
            braking_speed_reduction = 0.0
            start_braking_distance = 0.0
            start_braking_moments = []
            number_of_braking_moments = 0
            accelerating_distance = 0.0
            full_throttle_distance = 0.0
            last_throttle_level = 0.0
            last_distance = 0.0
            braking = False
            full_throttle = False
            for key, tel in lap_telemetry.iterrows():
                if tel['Brake'] and braking:
                    braking_distance += (tel['Distance'] - last_distance)
                elif tel['Brake']:
                    braking = True
                    braking_speed_start = tel['Speed']
                    start_braking_distance += tel['Distance']
                    start_braking_moments.append(tel['Distance'])
                    number_of_braking_moments += 1
                else:
                    if braking:
                        # end of braking
                        braking_speed_reduction += braking_speed_start - tel['Speed']
                    braking = False

                if 0 < last_throttle_level < tel['Throttle']:
                    accelerating_distance += (tel['Distance'] - last_distance)

                if tel['Throttle'] > 98 and full_throttle:
                    full_throttle_distance += (tel['Distance'] - last_distance)
                elif tel['Throttle'] > 98:
                    full_throttle = True
                else:
                    full_throttle = False

                last_throttle_level = float(tel['Throttle'])
                last_distance = float(tel['Distance'])
            drivers_df.at[index, 'Braking Distance'] = braking_distance
            drivers_df.at[index, 'Avg Braking Distance'] = (start_braking_distance / number_of_braking_moments)
            drivers_df.at[index, 'Avg Braking Speed Reduction'] = braking_speed_reduction
            drivers_df.at[index, 'Start Braking Moments'] = start_braking_moments
            drivers_df.at[index, 'Accelerating Distance'] = accelerating_distance
            drivers_df.at[index, 'Full Throttle Distance'] = full_throttle_distance

    showBrakingMoments(drivers_df)

    drivers_df = drivers_df.sort_values('Full Throttle Distance', ascending=False)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(drivers_df)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    drivers_df = drivers_df.sort_values('Braking Distance', ascending=False)
    ax1.bar(drivers_df['Driver'], drivers_df['Braking Distance'], label='Braking Distance',
            color=drivers_df['Team Color'])
    ax1.title.set_text('Total Braking Distance')
    ax1.tick_params(axis='x', rotation=35)

    drivers_df = drivers_df.sort_values('Avg Braking Speed Reduction', ascending=False)
    ax2.bar(drivers_df['Driver'], drivers_df['Avg Braking Speed Reduction'], label='Avg Braking Speed Reduction',
            color=drivers_df['Team Color'])
    ax2.title.set_text('Braking Speed Reduction')
    ax2.tick_params(axis='x', rotation=35)

    drivers_df = drivers_df.sort_values('Accelerating Distance', ascending=False)
    ax3.bar(drivers_df['Driver'], drivers_df['Accelerating Distance'], label='Accelerating Distance',
            color=drivers_df['Team Color'])
    ax3.title.set_text('Total Accelerating Distance')
    ax3.tick_params(axis='x', rotation=35)

    drivers_df = drivers_df.sort_values('Full Throttle Distance', ascending=False)
    ax4.bar(drivers_df['Driver'], drivers_df['Full Throttle Distance'], label='Full Throttle Distance',
            color=drivers_df['Team Color'])
    ax4.title.set_text('Full Throttle Distance')
    ax4.tick_params(axis='x', rotation=35)

    plt.suptitle(f"Fastest Lap Comparison \n "
                 f"{session.event['EventName']} {session.event.year} FP2")

    plt.show()


def showBrakingMoments(drivers_df):
    D = drivers_df.loc[:, 'Start Braking Moments'].to_numpy()
    x = []
    for i in range(1, 21):
        x.append(i * 2)

    driver_list = drivers_df.loc[:, 'Driver']
    print(driver_list)

    fig, ax = plt.subplots()
    ax.eventplot(D, lineoffsets=x, linewidth=0.75, color=drivers_df['Team Color'])
    plt.show()


def gearShiftsOnTrack(session, driver):
    lap = session.laps.pick_driver(driver).pick_fastest()
    tel = lap.get_telemetry()

    x = np.array(tel['X'].values)
    y = np.array(tel['Y'].values)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    gear = tel['nGear'].to_numpy().astype(float)

    cmap = cm.get_cmap('Paired')
    lc_comp = LineCollection(segments, norm=plt.Normalize(1, cmap.N + 1), cmap=cmap)
    lc_comp.set_array(gear)
    lc_comp.set_linewidth(4)

    plt.gca().add_collection(lc_comp)
    plt.axis('equal')
    plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

    title = plt.suptitle(
        f"Fastest Lap Gear Shift Visualization\n"
        f"{lap['Driver']} - {session.event['EventName']} {session.event.year}"
    )

    cbar = plt.colorbar(mappable=lc_comp, label="Gear", boundaries=np.arange(1, 10))
    cbar.set_ticks(np.arange(1.5, 9.5))
    cbar.set_ticklabels(np.arange(1, 9))

    plt.show()


def speedVisualOnTrackMap(session, driver):
    colormap = mpl.cm.plasma
    weekend = session.event
    lap = session.laps.pick_driver(driver).pick_fastest()

    # Get telemetry data
    x = lap.telemetry['X']  # values for x-axis
    y = lap.telemetry['Y']  # values for y-axis
    color = lap.telemetry['Speed']  # value to base color gradient on

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # We create a plot with title and adjust some setting to make it look good.
    fig, ax = plt.subplots(sharex=True, sharey=True, figsize=(12, 6.75))
    fig.suptitle(f'{weekend.EventName} {session.event.year} - {lap.Driver} - Speed', size=24, y=0.97)

    # Adjust margins and turn of axis
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.12)
    ax.axis('off')

    # After this, we plot the data itself.
    # Create background track line
    ax.plot(lap.telemetry['X'], lap.telemetry['Y'], color='black', linestyle='-', linewidth=16, zorder=0)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(color.min(), color.max())
    lc = LineCollection(segments, cmap=colormap, norm=norm, linestyle='-', linewidth=5)

    # Set the values used for colormapping
    lc.set_array(color)

    # Merge all line segments together
    line = ax.add_collection(lc)

    # Finally, we create a color bar as a legend.
    cbaxes = fig.add_axes([0.25, 0.05, 0.5, 0.05])
    normlegend = mpl.colors.Normalize(vmin=color.min(), vmax=color.max())
    legend = mpl.colorbar.ColorbarBase(cbaxes, norm=normlegend, cmap=colormap, orientation="horizontal")

    # Show the plot
    plt.show()


def qualifyingResultsOverview(session):
    drivers = pd.unique(session.laps['Driver'])
    list_fastest_laps = list()

    for drv in drivers:
        drvs_fastest_lap = session.laps.pick_driver(drv).pick_fastest()
        if not pd.isnull(drvs_fastest_lap['Time']):
            list_fastest_laps.append(drvs_fastest_lap)

    fastest_laps = Laps(list_fastest_laps).sort_values(by='LapTime').reset_index(drop=True)
    pole_lap = fastest_laps.pick_fastest()
    fastest_laps['LapTimeDelta'] = fastest_laps['LapTime'] - pole_lap['LapTime']

    team_colors = getTeamColors(fastest_laps)

    fig, ax = plt.subplots()
    ax.barh(fastest_laps.index, fastest_laps['LapTimeDelta'],
            color=team_colors, edgecolor='grey')
    ax.set_yticks(fastest_laps.index)
    ax.set_yticklabels(fastest_laps['Driver'])

    # show fastest at the top
    ax.invert_yaxis()

    # draw vertical lines behind the bars
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, which='major', linestyle='--', color='black', zorder=-1000)

    lap_time_string = strftimedelta(pole_lap['LapTime'], '%m:%s.%ms')

    plt.suptitle(f"{session.event['EventName']} {session.event.year} Qualifying\n"
                 f"Fastest Lap: {lap_time_string} ({pole_lap['Driver']})")

    plt.show()


def qualifyingDifferenceBetweenFastestLapAndFastestSectors(session):
    drivers = pd.unique(session.laps['Driver'])
    list_fastest_laps = list()

    for drv in drivers:
        fastest_sectors = [None] * 3

        for index, lap in session.laps.pick_driver(drv).iterlaps():
            if (fastest_sectors[0] is None or fastest_sectors[0] > lap['Sector1Time']) and not pd.isnull(lap['Sector1Time']):
                fastest_sectors[0] = lap['Sector1Time']
            if (fastest_sectors[1] is None or fastest_sectors[1] > lap['Sector2Time']) and not pd.isnull(lap['Sector2Time']):
                fastest_sectors[1] = lap['Sector2Time']
            if (fastest_sectors[2] is None or fastest_sectors[2] > lap['Sector3Time']) and not pd.isnull(lap['Sector3Time']):
                fastest_sectors[2] = lap['Sector3Time']
        drvs_fastest_lap = session.laps.pick_driver(drv).pick_fastest()
        drvs_fastest_lap['fastest_lap_sectors'] = fastest_sectors[0] + fastest_sectors[1] + fastest_sectors[2]
        drvs_fastest_lap['FastestSector1Time'] = fastest_sectors[0]
        drvs_fastest_lap['FastestSector2Time'] = fastest_sectors[1]
        drvs_fastest_lap['FastestSector3Time'] = fastest_sectors[2]

        if not pd.isnull(drvs_fastest_lap['Time']):
            list_fastest_laps.append(drvs_fastest_lap)

    fastest_lap_sectors = Laps(list_fastest_laps).sort_values(by='fastest_lap_sectors').reset_index(drop=True)
    fastest_lap = fastest_lap_sectors.iloc[0]['fastest_lap_sectors']
    fastest_lap_sectors['LapTimeDelta'] = fastest_lap_sectors['fastest_lap_sectors'] - fastest_lap

    sector_times = []
    for sector in range(1, 4):
        sector_times.append(Laps(list_fastest_laps).sort_values(by='FastestSector' + str(sector) + 'Time').reset_index(drop=True))
        fastest_sector = sector_times[sector - 1].iloc[0]['FastestSector' + str(sector) + 'Time']
        sector_times[sector - 1]['Sector' + str(sector) + 'TimeDelta'] = sector_times[sector - 1]['FastestSector' + str(sector) + 'Time'] - fastest_sector

    team_colors = getTeamColors(fastest_lap_sectors)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4)
    ax1.barh(fastest_lap_sectors.index, fastest_lap_sectors['LapTimeDelta'], color=team_colors, edgecolor='grey')
    ax1.set_yticks(fastest_lap_sectors.index)
    ax1.set_yticklabels(fastest_lap_sectors['Driver'])
    ax1.invert_yaxis()
    ax1.set_axisbelow(True)
    ax1.xaxis.grid(True, which='major', linestyle='--', color='black', zorder=-1000)
    ax1.title.set_text('Overall Lap')
    ax1.tick_params(axis='x', rotation=35)

    team_colors = getTeamColors(sector_times[0])
    ax2.barh(sector_times[0].index, sector_times[0]['Sector1TimeDelta'], color=team_colors, edgecolor='grey')
    ax2.set_yticks(sector_times[0].index)
    ax2.set_yticklabels(sector_times[0]['Driver'])
    ax2.invert_yaxis()
    ax2.set_axisbelow(True)
    ax2.xaxis.grid(True, which='major', linestyle='--', color='black', zorder=-1000)
    ax2.title.set_text('Best Sector 1')
    ax2.tick_params(axis='x', rotation=35)

    team_colors = getTeamColors(sector_times[1])
    ax3.barh(sector_times[1].index, sector_times[1]['Sector2TimeDelta'], color=team_colors,
             edgecolor='grey')
    ax3.set_yticks(sector_times[1].index)
    ax3.set_yticklabels(sector_times[1]['Driver'])
    ax3.invert_yaxis()
    ax3.set_axisbelow(True)
    ax3.xaxis.grid(True, which='major', linestyle='--', color='black', zorder=-1000)
    ax3.title.set_text('Best Sector 2')
    ax3.tick_params(axis='x', rotation=35)

    team_colors = getTeamColors(sector_times[2])
    ax4.barh(sector_times[2].index, sector_times[2]['Sector3TimeDelta'], color=team_colors,
             edgecolor='grey')
    ax4.set_yticks(sector_times[2].index)
    ax4.set_yticklabels(sector_times[2]['Driver'])
    ax4.invert_yaxis()
    ax4.set_axisbelow(True)
    ax4.xaxis.grid(True, which='major', linestyle='--', color='black', zorder=-1000)
    ax4.title.set_text('Best Sector 3')
    ax4.tick_params(axis='x', rotation=35)

    plt.suptitle(f"{session.event['EventName']} {session.event.year} Qualifying\n")
    plt.show()


def getStintLapTimesByTyreCompound(session, driver, compound):
    drv_lap_times = []
    laps_info = ""
    team_color = ""
    current_stint = 0
    for index, lap in session.laps.pick_driver(driver).iterlaps():
        if lap['Compound'] == compound and not pd.isnull(lap['LapTime']):
            laps_info = lap['Driver'] + ' Stint ' + str(lap['Stint'])
            team_color = fastf1.plotting.team_color(lap['Team'])
            drv_lap_times.append(lap['LapTime'])
        if current_stint > 0 and current_stint != lap['Stint']:
            return drv_lap_times, laps_info, team_color
        current_stint = lap['Stint']
    return drv_lap_times, laps_info, team_color


def lapTimesByTyreCompound(session, compound, drivers=0):
    driver_stint_laps = []
    driver_stint_labels = []

    if drivers == 0:
        drivers = pd.unique(session.laps['Driver'])

    fig, ax = plt.subplots()

    for drv in drivers:
        drv_lap_times, laps_info, team_color = getStintLapTimesByTyreCompound(session, drv, compound)
        drv_lap_times = []
        laps_info = ""
        team_color = ""
        current_stint = 0
        for index, lap in session.laps.pick_driver(drv).iterlaps():
            if lap['Compound'] == compound and not pd.isnull(lap['LapTime']):
                print(lap['Stint'], lap['Compound'], lap['LapTime'], lap['TrackStatus'])
                laps_info = lap['Driver'] + ' Stint ' + str(current_stint)
                team_color = fastf1.plotting.team_color(lap['Team'])
                drv_lap_times.append(lap['LapTime'])
                if current_stint > 0 and current_stint != lap['Stint']:
                    driver_stint_laps.append(drv_lap_times)
                    driver_stint_labels.append(laps_info)
                    plt.plot(range(1, len(drv_lap_times)+1), drv_lap_times, label=laps_info)
                    drv_lap_times = []
                current_stint = lap['Stint']
        if drv_lap_times:
            driver_stint_laps.append(drv_lap_times)
            driver_stint_labels.append(laps_info)
            plt.plot(range(1, len(drv_lap_times)+1), drv_lap_times, label=laps_info)

    # box = ax.boxplot(driver_stint_laps, 0, '')
    # ax.set_xticklabels(driver_stint_labels)

    edge_color = 'blue'
    fill_color = 'cyan'

    ax.set_xlabel('Lap Number')
    ax.set_ylabel('Lap Time')
    ax.invert_yaxis()
    ax.legend()
    # ax.tick_params(axis='x', rotation=90)
    plt.suptitle(f"Lap Comparison \n "
                 f"{session.event['EventName']} {session.event.year} Race")

    plt.show()


def raceLapAnimationReview(session, drivers=0):
    if drivers == 0:
        drivers = pd.unique(session.laps['Driver'])

    fig, ax = plt.subplots()

    for drv in drivers:
        driver_all_laps = pd.DataFrame(columns=['SessionTime', 'LapNumber', 'DriverAhead', 'DistanceToDriverAhead'])
        for index, lap in session.laps.pick_driver(drv).iterlaps():
            driver_telemetry = lap.get_telemetry()
            driver_lap_data = driver_telemetry[['SessionTime', 'DriverAhead', 'DistanceToDriverAhead']]
            driver_lap_data.loc[:, 'LapNumber'] = lap['LapNumber']
            driver_all_laps = pd.concat([driver_all_laps, driver_lap_data], axis=0, ignore_index=True)
        plt.plot(range(1, len(driver_all_laps) + 1), driver_all_laps['DistanceToDriverAhead'], label=drv)

    plt.legend()
    plt.show()

    # print(driver_laps_all.head())

    # distanceToDriverAnimation(driver_laps_all)


def distanceToDriverAnimation(driver_laps_all):
    distance_to_next_driver = driver_laps_all['DistanceToDriverAhead']

    fig, ax = plt.subplots()
    fig.suptitle('Distance to Next Driver', fontsize=14)

    bars = ax.barh(['RIC'], [0], alpha=0.75)
    ax.set_xlim([0, max(distance_to_next_driver)])
    # ax.invert_xaxis()

    def animate(i):
        if i in driver_laps_all.index:
            bars[0].set_width(distance_to_next_driver[i])
            fig.suptitle('Distance to Next Driver - lap ' + str(driver_laps_all.loc[i, 'LapNumber']))

    # calling the animation function
    anim = FuncAnimation(fig, animate, frames=len(distance_to_next_driver), interval=2, repeat=False)
    plt.show()
