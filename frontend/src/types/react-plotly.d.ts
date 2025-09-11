declare module 'react-plotly.js' {
  import { Component } from 'react';
  
  interface PlotParams {
    data: any[];
    layout?: any;
    config?: any;
    frames?: any[];
    revision?: number;
    onInitialized?: (figure: any, graphDiv: HTMLElement) => void;
    onUpdate?: (figure: any, graphDiv: HTMLElement) => void;
    onPurge?: (figure: any, graphDiv: HTMLElement) => void;
    onError?: (err: any) => void;
    onSelected?: (eventData: any) => void;
    onDeselect?: () => void;
    onHover?: (eventData: any) => void;
    onUnhover?: (eventData: any) => void;
    onClick?: (eventData: any) => void;
    onClickAnnotation?: (eventData: any) => void;
    onAnimatingFrame?: (eventData: any) => void;
    onAnimated?: () => void;
    onTransitioning?: () => void;
    onTransitioned?: () => void;
    onRelayout?: (eventData: any) => void;
    onRedraw?: () => void;
    onAfterExport?: () => void;
    onAfterPlot?: () => void;
    onAnimationInterrupted?: () => void;
    onAutoSize?: () => void;
    onBeforeExport?: () => void;
    onButtonClicked?: (eventData: any) => void;
    onDoubleClick?: () => void;
    onFramework?: () => void;
    onLegendClick?: (eventData: any) => boolean;
    onLegendDoubleClick?: (eventData: any) => boolean;
    onRestyle?: (eventData: any) => void;
    onSliderChange?: (eventData: any) => void;
    onSliderEnd?: (eventData: any) => void;
    onSliderStart?: (eventData: any) => void;
    onSunburstClick?: (eventData: any) => void;
    onTreemapClick?: (eventData: any) => void;
    onWebGlContextLost?: () => void;
    divId?: string;
    className?: string;
    style?: React.CSSProperties;
    debug?: boolean;
    useResizeHandler?: boolean;
  }
  
  export default class Plot extends Component<PlotParams> {}
}
