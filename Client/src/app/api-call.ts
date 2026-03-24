import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { Metrics } from './model/metrics';
import { RegressionPrediction } from './model/regression-predict';
import { ClassifierPrediction } from './model/classifier_prediction';

@Injectable({
  providedIn: 'root',
})
export class ApiCall {
  private static readonly BASE_URL: string = 'http://localhost:8000/v1/machineLearn';
  constructor(private http: HttpClient){}
  //-------------------------------- BANK ----------------------------------------//

  getMetrics(): Observable<Metrics>{
    const api = ApiCall.BASE_URL + '/metrics';
    return this.http.get<Metrics>(api);
  }

  analysis(): Observable<string>{
    const api = ApiCall.BASE_URL + '/analysis';
    return this.http.get(api, {responseType: 'text'});
  }

  predictRegression(data: RegressionPrediction): Observable<number>{
    const api = ApiCall.BASE_URL + '/regression/predict';
    return this.http.post<number>(api, data);
  }

  predictClassifier(data: ClassifierPrediction): Observable<number>{
    const api = ApiCall.BASE_URL + '/classifier/predict';
    return this.http.post<number>(api, data);
  }
}
