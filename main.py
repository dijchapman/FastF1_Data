import fastf1
import fastf1.plotting
import pandas as pd
import f1visuals


def main():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 10000)

    fastf1.Cache.enable_cache('C:/Users/dave/PycharmProjects/fastf1_data')
    fastf1.plotting.setup_mpl()
    session = fastf1.get_session(2022, 'Hungary', 'R')
    session.load()

    driver_code = 'VET'

    driver_laps = session.laps.pick_driver(driver_code)

    # for index, lap in driver_laps.iterlaps():
        # print("Lap #" + str(lap['LapNumber']) + " = " + str(lap['LapTime']) + " on " + lap['Compound'] + " compound")

    # f1visuals.lapTelemetryData(session, 'RIC')
    # f1visuals.driverTelemetryComparisons(session)
    # f1visuals.gearShiftsOnTrack(session, 'RUS')
    # f1visuals.speedVisualOnTrackMap(session, 'RUS')
    # f1visuals.qualifyingResultsOverview(session)
    # f1visuals.qualifyingDifferenceBetweenFastestLapAndFastestSectors(session)
    # f1visuals.lapTimesByTyreCompound(session, 'SOFT', drivers=['RIC', 'NOR'])
    # f1visuals.lapTimesByTyreCompound(session, 'MEDIUM', drivers=['RIC', 'NOR'])
    f1visuals.lapTimesByTyreCompound(session, 'HARD', drivers=['RIC', 'NOR'])
    # f1visuals.raceLapAnimationReview(session, drivers=['RIC', 'NOR'])


if __name__ == "__main__":
    # calling main function
    main()
