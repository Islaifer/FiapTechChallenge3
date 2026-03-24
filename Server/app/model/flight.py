import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from typing import Optional

class FlightClassifier(BaseModel):
    distance: float = Field(0)
    diverted: int = Field(0)
    month: int = Field(0)
    day: int = Field(0)
    day_of_week: int = Field(0)
    scheduled_departure_hour: int = Field(0)
    origin: Optional[str] = Field('')
    destination: Optional[str] = Field('')
    airline: Optional[str] = Field('')

    def to_dataframe(self):
        dep_minutes = (self.scheduled_departure_hour // 100) * 60 + (self.scheduled_departure_hour % 100)

        dep_sin = np.sin(2 * np.pi * dep_minutes / 1440)
        dep_cos = np.cos(2 * np.pi * dep_minutes / 1440)

        season = ''
        if (self.month == 2 & self.day >= 20) | (self.month == 4 | self.month ==5) | (self.month == 6 & self.day <= 20):
            season = 'Spring'
        elif (self.month == 6 & self.day >= 21) | (self.month == 7 | self.month ==8) | (self.month == 9 & self.day <= 22):
            season = 'Summer'
        elif (self.month == 9 & self.day >= 23) | (self.month == 10 | self.month ==11) | (self.month == 12 & self.day <= 20):
            season = 'Fall'
        else:
            season = 'Winter'

        month_sin = np.sin(2 * np.pi * self.month / 12)
        month_cos = np.cos(2 * np.pi * self.month / 12)

        days_in_month = 0

        if self.month == 2:
            days_in_month = 28
        elif self.month == 4 | self.month == 6 | self.month == 9 | self.month == 11:
            days_in_month = 30
        else:
            days_in_month = 31

        day_sin = np.sin(2 * np.pi * self.day / days_in_month)
        day_cos = np.cos(2 * np.pi * self.day / days_in_month)

        day_of_week_sin = np.sin(2 * np.pi * self.day_of_week / 7)
        day_of_week_cos = np.cos(2 * np.pi * self.day_of_week / 7)

        return pd.DataFrame([{
            "DISTANCE": self.distance,
            "DIVERTED": self.diverted,
            "DEP_SIN": dep_sin,
            "DEP_COS": dep_cos,
            "MONTH_SIN": month_sin,
            "MONTH_COS": month_cos,
            "DAY_SIN": day_sin,
            "DAY_COS": day_cos,
            "DAY_OF_WEEK_SIN": day_of_week_sin,
            "DAY_OF_WEEK_COS": day_of_week_cos,
            "ORIGIN": self.origin,
            "DESTINATION": self.destination,
            "AIRLINE": self.airline,
            "SEASON": season
        }])

class FlightRegression(BaseModel):
    distance: float = Field(0)
    diverted: int = Field(0)
    month: int = Field(0)
    day: int = Field(0)
    day_of_week: int = Field(0)
    scheduled_departure_hour: int = Field(0)
    origin: Optional[str] = Field('')
    destination: Optional[str] = Field('')
    airline: Optional[str] = Field('')
    has_weather_delay: int = Field(0)
    has_airline_delay: int = Field(0)
    has_air_system_delay: int = Field(0)
    has_security_delay: int = Field(0)
    has_late_aircraft_delay: int = Field(0)
    
    def to_dataframe(self):
        dep_minutes = (self.scheduled_departure_hour // 100) * 60 + (self.scheduled_departure_hour % 100)

        dep_sin = np.sin(2 * np.pi * dep_minutes / 1440)
        dep_cos = np.cos(2 * np.pi * dep_minutes / 1440)

        season = ''
        if (self.month == 2 & self.day >= 20) | (self.month == 4 | self.month ==5) | (self.month == 6 & self.day <= 20):
            season = 'Spring'
        elif (self.month == 6 & self.day >= 21) | (self.month == 7 | self.month ==8) | (self.month == 9 & self.day <= 22):
            season = 'Summer'
        elif (self.month == 9 & self.day >= 23) | (self.month == 10 | self.month ==11) | (self.month == 12 & self.day <= 20):
            season = 'Fall'
        else:
            season = 'Winter'

        month_sin = np.sin(2 * np.pi * self.month / 12)
        month_cos = np.cos(2 * np.pi * self.month / 12)

        days_in_month = 0

        if self.month == 2:
            days_in_month = 28
        elif self.month == 4 | self.month == 6 | self.month == 9 | self.month == 11:
            days_in_month = 30
        else:
            days_in_month = 31

        day_sin = np.sin(2 * np.pi * self.day / days_in_month)
        day_cos = np.cos(2 * np.pi * self.day / days_in_month)

        day_of_week_sin = np.sin(2 * np.pi * self.day_of_week / 7)
        day_of_week_cos = np.cos(2 * np.pi * self.day_of_week / 7)

        return pd.DataFrame([{
            "DISTANCE": self.distance,
            "DIVERTED": self.diverted,
            "DEP_SIN": dep_sin,
            "DEP_COS": dep_cos,
            "MONTH_SIN": month_sin,
            "MONTH_COS": month_cos,
            "DAY_SIN": day_sin,
            "DAY_COS": day_cos,
            "DAY_OF_WEEK_SIN": day_of_week_sin,
            "DAY_OF_WEEK_COS": day_of_week_cos,
            "ORIGIN": self.origin,
            "DESTINATION": self.destination,
            "AIRLINE": self.airline,
            "HAS_WEATHER_DELAY": self.has_weather_delay,
            "HAS_AIRLINE_DELAY": self.has_airline_delay,
            "HAS_AIR_SYSTEM_DELAY": self.has_air_system_delay,
            "HAS_SECURITY_DELAY": self.has_security_delay,
            "HAS_LATE_AIRCRAFT_DELAY": self.has_late_aircraft_delay,
            "SEASON": season
        }])