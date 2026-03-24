export class RegressionPrediction {
    distance: number;
    diverted: number;
    month: number;
    day: number;
    day_of_week: number;
    scheduled_departure_hour: number;
    origin: string
    destination: string
    airline: string
    has_weather_delay: number;
    has_airline_delay: number;
    has_air_system_delay: number;
    has_security_delay: number;
    has_late_aircraft_delay: number;

    constructor(){
        this.distance = 0;
        this.diverted = 0;
        this.month = 0;
        this.day = 0;
        this.day_of_week = 0;
        this.scheduled_departure_hour = 0;
        this.origin = "";
        this.destination = "";
        this.airline = "";
        this.has_weather_delay = 0;
        this.has_airline_delay = 0;
        this.has_air_system_delay = 0;
        this.has_security_delay = 0;
        this.has_late_aircraft_delay = 0;
    }
}