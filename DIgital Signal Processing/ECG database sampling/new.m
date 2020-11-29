function varargout = new(varargin)
% NEW MATLAB code for new.fig
%      NEW, by itself, creates a new NEW or raises the existing
%      singleton*.
%
%      H = NEW returns the handle to a new NEW or the handle to
%      the existing singleton*.
%
%      NEW('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in NEW.M with the given input arguments.
%
%      NEW('Property','Value',...) creates a new NEW or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before new_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to new_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help new

% Last Modified by GUIDE v2.5 19-Feb-2019 03:37:02

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @new_OpeningFcn, ...
                   'gui_OutputFcn',  @new_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before new is made visible.
function new_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to new (see VARARGIN)

% Choose default command line output for new
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes new wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = new_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in Browser.
function Browser_Callback(hObject, eventdata, handles)
% hObject    handle to Browser (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global t
global v
global k
global n
global mi 
global m

cla(handles.axes1)
axes(handles.axes1)
file_selected = uigetfile({'*.csv'},'File selector');
set(handles.FileName,'string',file_selected);
SignalData = csvread(file_selected);
k=1;
t = SignalData(:,1);
v = SignalData(:,2);
n =length(t) ;
m = max(v);
mi = min(v);
xlabel('Time(Sec)');
ylabel('Voltage');


% --- Executes on button press in Control_Button.
function Control_Button_Callback(hObject, eventdata, handles)
% hObject    handle to Control_Button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of Control_Button
global t;
global v;
global k;
global n;
global i;
i=1;
h=n/10;      % 10 Regions
control = get(hObject,'Value');
s=150;
y1=-inf;
y2=inf;
while i<10        %Regions Counter
while control && k <= h
       plot(t,v,'g','LineWidth',2);
       axis([t(k) t(k+s) y1 y2]);
       drawnow
       pause (0.03)
       k=k+1;
       control = get(hObject,'Value');      
end 
   set(handles.text5, 'String', num2str(i)); 
   set(handles.slider1,'Value',i);
   i=i+1;
   h=h+100;
 end 


function FileName_Callback(hObject, eventdata, handles)
% hObject    handle to FileName (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of FileName as text
%        str2double(get(hObject,'String')) returns contents of FileName as a double


% --- Executes during object creation, after setting all properties.
function FileName_CreateFcn(hObject, eventdata, handles)
% hObject    handle to FileName (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on slider movement.
function slider1_Callback(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
global mi 
global m 
axes(handles.axes1);

slider = get (handles.slider1, 'Value') ;
if slider ==0 
      set(gca,'Xlim',[0 1],'Ylim',[mi m]);
elseif slider == 1 
     set ( gca , 'XLim' ,[1  2] , 'YLim' , [mi m]  ) ;
     set(handles.text5,'String','1');
 elseif slider == 2 
     set ( gca , 'XLim' ,[2  3]  , 'YLim' , [mi m] );
      set(handles.text5,'String',' 2');
 elseif slider == 3
     set ( gca , 'XLim' , [3  4]  , 'YLim' ,[mi m] );
      set(handles.text5,'String','3');
 elseif slider == 4
     set ( gca , 'XLim' ,[4  5]  , 'YLim' , [mi m] ) ;
      set(handles.text5,'String',' 4');
 elseif slider == 5
     set ( gca , 'XLim' ,[5 6 ]  , 'YLim' , [mi m]) ;
      set(handles.text5,'String',' 5');
 elseif slider == 6
     set ( gca , 'XLim' ,[6 7 ]  , 'YLim' ,[mi m] );
      set(handles.text5,'String',' 6');
 elseif slider == 7
     set ( gca , 'XLim', [7 8] , 'YLim' ,[mi m]  );
      set(handles.text5,'String',' 7');
 elseif slider == 8
     set ( gca , 'XLim' ,[8 9] , 'YLim' ,[mi m] ) ;
      set(handles.text5,'String',' 8');
 elseif slider == 9
     set ( gca , 'XLim' , [9 10] , 'YLim' ,[mi m] ) ;
      set(handles.text5,'String',' 9');
 elseif slider== 10 
         set ( gca , 'XLim' ,[9  10] , 'YLim' ,[mi m]  );
          set(handles.text5,'String','10');   
 end 

% --- Executes during object creation, after setting all properties.
function slider1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- If Enable == 'on', executes on mouse press in 5 pixel border.
% --- Otherwise, executes on mouse press in 5 pixel border or over slider1.
function slider1_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)