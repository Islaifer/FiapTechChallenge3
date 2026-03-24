export class Metrics {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    rmse: number;
    mae: number;
    silhouette: number;

    constructor(){
        this.accuracy = 0;
        this.precision = 0;
        this.recall = 0;
        this.f1 = 0;
        this.rmse = 0;
        this.mae = 0;
        this.silhouette = 0
    }
}