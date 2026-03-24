export class ClassifierPrediction {
    distance: number;
    diverted: number;
    month: number;
    day: number;
    day_of_week: number;
    scheduled_departure_hour: number;
    origin: string
    destination: string
    airline: string

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
    }
}