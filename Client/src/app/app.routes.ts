import { Routes } from '@angular/router';
import { Dashboard } from './dashboard/dashboard/dashboard';

export const routes: Routes = [
    { path: '', component: Dashboard},
    //{
    //    path: 'system',
    //    canActivate: [authGuard],
    //    loadComponent: () => import('./system/main-page/main-page').then(p => p.MainPage),
    //    children: [
    //        {
    //            path: 'financial',
    //            loadComponent: () => import('./system/financial/financial').then(p => p.Financial)
    //        },
    //        {
    //            path: 'dashboard',
    //            loadComponent: () => import('./system/dashboard/dashboard').then(p => p.Dashboard)
    //        }
    //    ]
    //}
];
