import { subscribeOn } from 'rxjs/operators';
import {Features} from './features'
import { SubObject } from './sub-object';

export class MainObject {
    confidence:number;
    text:string;
    features:Features[];
    subObject:SubObject[];
}
