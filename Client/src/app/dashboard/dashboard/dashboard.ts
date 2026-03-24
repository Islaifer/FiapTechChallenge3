import { DatePipe, CommonModule } from '@angular/common';
import { Component, inject, OnInit } from '@angular/core';
import { FormControl, FormGroup, FormsModule, ReactiveFormsModule, Validators } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatIconModule } from '@angular/material/icon';
import { MatListModule } from '@angular/material/list';
import { MatMenuModule } from '@angular/material/menu';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatSelectModule } from '@angular/material/select';
import { ApiCall } from '../../api-call';
import { MatSnackBar } from '@angular/material/snack-bar';
import { Metrics } from '../../model/metrics';
import { RegressionPrediction } from '../../model/regression-predict';
import { ClassifierPrediction } from '../../model/classifier_prediction';
import { MatInputModule } from '@angular/material/input';
import { MatDialogContent, MatDialogTitle } from '@angular/material/dialog';

@Component({
  selector: 'app-dashboard',
  imports: [
    MatCardModule,
    MatButtonModule,
    MatIconModule,
    MatMenuModule,
    MatFormFieldModule,
    FormsModule,
    MatSelectModule,
    MatListModule,
    CommonModule,
    ReactiveFormsModule,
    MatInputModule
  ],
  templateUrl: './dashboard.html',
  styleUrl: './dashboard.scss',
})
export class Dashboard implements OnInit {
  metrics: Metrics;
  alreadyAnalysis: boolean = false;
  private _snackBar = inject(MatSnackBar);
  regressionFormGroup: FormGroup;
  classifierFormGroup: FormGroup;

  constructor(
    private apiCaller: ApiCall
  ){
    this.regressionFormGroup = new FormGroup({
      distance: new FormControl(0, Validators.required),
      diverted: new FormControl(0, Validators.required),
      month: new FormControl(0, Validators.required),
      day: new FormControl(0, Validators.required),
      day_of_week: new FormControl(0, Validators.required),
      scheduled_departure_hour: new FormControl(0, Validators.required),
      origin: new FormControl('', Validators.required),
      destination: new FormControl('', Validators.required),
      airline: new FormControl('', Validators.required),
      has_weather_delay: new FormControl(0, Validators.required),
      has_airline_delay: new FormControl(0, Validators.required),
      has_air_system_delay: new FormControl(0, Validators.required),
      has_security_delay: new FormControl(0, Validators.required),
      has_late_aircraft_delay: new FormControl(0, Validators.required)
    });

    this.classifierFormGroup = new FormGroup({
      distance: new FormControl(0, Validators.required),
      diverted: new FormControl(0, Validators.required),
      month: new FormControl(0, Validators.required),
      day: new FormControl(0, Validators.required),
      day_of_week: new FormControl(0, Validators.required),
      scheduled_departure_hour: new FormControl(0, Validators.required),
      origin: new FormControl('', Validators.required),
      destination: new FormControl('', Validators.required),
      airline: new FormControl('', Validators.required),
    });
    this.metrics = new Metrics();
  }

  ngOnInit(): void {
    this.getMetrics();
  }

  getMetrics(){
    this.apiCaller.getMetrics().subscribe({
      next: p => {
        this.metrics = p;
        if(this.metrics.accuracy != 0)
          this.alreadyAnalysis = true;
      }
    });
  }

  analysis(){
    this.apiCaller.analysis().subscribe({
      next: p => {
        this._snackBar.open(p, 'Ok', { duration: 3000 });
      }
    });
  }

  resultRegression: number = 0;
  predictRegression(){
    this.regressionFormGroup.markAllAsTouched();

    if(this.regressionFormGroup.valid){
      const values = this.regressionFormGroup.value;

      const regression = new RegressionPrediction();
      regression.distance = values["distance"];
      regression.diverted = values["diverted"];
      regression.month = values["month"];
      regression.day = values["day"];
      regression.day_of_week = values["day_of_week"];
      regression.scheduled_departure_hour = values["scheduled_departure_hour"];
      regression.origin = values["origin"];
      regression.destination = values["destination"];
      regression.airline = values["airline"];
      regression.has_weather_delay = values["has_weather_delay"];
      regression.has_airline_delay = values["has_airline_delay"];
      regression.has_air_system_delay = values["has_air_system_delay"];
      regression.has_security_delay = values["has_security_delay"];
      regression.has_late_aircraft_delay = values["has_late_aircraft_delay"];
    
      this.apiCaller.predictRegression(regression).subscribe({
        next: p => {
          this.resultRegression = p;
        }
      });
    }
  }

  classifierResult: number = -1;
  predictClassifier(){
    this.classifierFormGroup.markAllAsTouched();

    if(this.classifierFormGroup.valid){
      const values = this.classifierFormGroup.value;

      const classifier = new ClassifierPrediction();
      classifier.distance = values["distance"];
      classifier.diverted = values["diverted"];
      classifier.month = values["month"];
      classifier.day = values["day"];
      classifier.day_of_week = values["day_of_week"];
      classifier.scheduled_departure_hour = values["scheduled_departure_hour"];
      classifier.origin = values["origin"];
      classifier.destination = values["destination"];
      classifier.airline = values["airline"];
    
      this.apiCaller.predictClassifier(classifier).subscribe({
        next: p => {
          this.classifierResult = p;
        }
      });
    }
  }

  isInvalidFieldRegression(fieldName: string) : boolean{
    const field = this.regressionFormGroup.get(fieldName);
    return field?.invalid && field?.touched && field?.errors?.['required']
  }

  isInvalidFieldClassifier(fieldName: string) : boolean{
    const field = this.classifierFormGroup.get(fieldName);
    return field?.invalid && field?.touched && field?.errors?.['required']
  }
}
