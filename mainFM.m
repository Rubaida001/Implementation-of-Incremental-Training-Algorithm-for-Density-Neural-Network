clear all; 
clc;
T  = 500; 
b = 0.1; 
c = 0.2; 
tau = 17;
% Initialize values with random function
random_value= [ 0.401310 0.953833 0.174821 0.572708 0.971513 0.109872 0.388265 ...
    0.942936 0.213617 0.666899 0.881914 0.413442 0.962755 0.142354 ...
    0.484694 0.991570 0.033185 0.127373 0.441263 0.978804]'; 

% Mackay-Glass time series generation
for t = 20:T+49
    random_value(t+1) = random_value(t)+c*random_value(t-tau)/(1+random_value(t-tau).^10)-b*random_value(t);
end
random_value(1:50) = [];

%Normalize values
nrmY=random_value(50:T);
ymin=min(nrmY(:)); 
ymax=max(nrmY(:)); 
relative_val=2.0*((nrmY-ymin)/(ymax-ymin)-0.5);

% create a matrix of lagged values for a time series vector
Ss=relative_val';
input_dim=10; % input dimension
output_dim=length(Ss)-input_dim; % output dimension
for i=1:output_dim
   y(i)=Ss(i+input_dim);
   for j=1:input_dim
      x(i,j) = Ss(i-j+input_dim); %x(i,idim-j+1) = Ss(i-j+idim);
   end
end

Patterns = x'; 
Desired = y; 
NHIDDEN = 3; 
NHIDDENS = 3; 
[NINPUTS,NPATS] = size(Patterns); 

Inputs1= [Patterns;ones(1,NPATS)]; %Inputs1 = [ones(1,NPATS); Patterns];

wm_i = 0.5*(rand(NHIDDEN,1+NINPUTS)-0.5); 
wm_o = 0.5*(rand(1,1+NHIDDEN)-0.5);
wv_i = 0.5*(rand(NHIDDENS,1+NINPUTS)-0.5);
wv_o = 0.5*(rand(1,1+NHIDDENS)-0.5);

% Phase 1
learning_rate_m = 0.0001;
for epoch = 1:1500
% Forward propagation (mean network):
    sum1 =wm_i*Inputs1; 
    hidden =tanh(sum1);
    sum2v =wm_o*[hidden;ones(1,NPATS)]; 
    out =sum2v;
% Backpropagation (mean network):
    error1 = Desired-out; sse = sum(sum(error1.^2));
    sig = 1;% mean(error.^2);%%%%%%%%%%%%%%%
    bout = error1./sig;
    bp = (wm_o'*bout); bh = (1.0-hidden).^2.*bp(1:end-1,:);
    % Computing weight deltas (mean network):
    dW2 = bout*[hidden;ones(1,NPATS)]'; dW1 = bh*Inputs1';
    % Updating the weights (mean network):
    wm_o = wm_o+learning_rate_m*dW2; wm_i = wm_i+learning_rate_m*dW1;
end
ph1_mean=out;

% Phase 2
learning_rate_v = 0.000001;%0.00001;
for epoch = 1:1500
% Forward propagation (variance network):
    sum1s =wv_i*Inputs1; hiddens =tanh(sum1s);
    sum2s =wv_o*[hiddens;ones(1,NPATS)];
    outs = exp(sum2s); sig = outs;
% Backpropagation (variance network):
    error = Desired-outs; sse = sum(sum(error.^2));
    bouts = ((error.*error)./sig-1.0)/2;
    bps = (wv_o'*bouts); bhs = (1.0-hiddens).^2.*bps(1:end-1,:);
% Computing weight deltas (variance network):
    dW2s = bouts*[hiddens;ones(1,NPATS)]'; dW1s = bhs*Inputs1';
% Updating the weights (variance network):
    wv_o = wv_o+learning_rate_v*dW2s; wv_i = wv_i+learning_rate_v*dW1s;
end

% Phase 3
sig = mean(error1.^2);
for epoch = 1:3000
    for i = 1:length(Inputs1)
        %% mean network
        %% forward propagation
        input = Inputs1(:,i);
        hidden=zeros(size(wm_i,1),size(input,2));
        %disp(input);
        for j = 1:size(wm_i,1)
            sum1 = 0;
            for k = 1:length(wm_i)
                sum1 = sum1 + wm_i(j,k)*input(k);
            end
            hidden(j,1) = tanh(sum1);
        end
        sum2 =0;
        mat = vertcat(hidden,ones(1,1));
        for j = 1:length(wm_o)
            sum2 = sum2 + wm_o(j)*mat(j);
        end
        out(i) = sum2;
        
        %% back propagation
        error = Desired(i) - out(i);
        sse_m = sum(sum(error.^2)); fprintf(' sse_m = %f\n ',sse_m);
        bout = error/sig;    
        bp = zeros(length(wm_o),size(wm_o,1));
        for j = 1:size(wm_o,1)
            for k = 1:length(wm_o)
                bp(k,j) = wm_o(j,k)*bout;
            end
        end
        bh=zeros(size(hidden,1),size(hidden,2));
        for j = 1:size(hidden,1)
            bh(j,:)= (1-hidden(j,:).^2).*bp(j,:);
        end     
         
        %% weight update for input to hidden network 
         for row = 1:size(bh,1)
             for col = 1:size(input,1)
              wm_i(row,col) = wm_i(row,col)+learning_rate_m*bh(row,:)*input(col,:); 
             end
         end
         
         %% weight update for hidden to output network 
        hidden_to_out = [hidden;ones(1,1)]; 
        for j = 1:length(hidden_to_out)
            wm_o(j) = wm_o(j)+learning_rate_m*bout*hidden_to_out(j);
        end    
                
 %% variance network
        %% forward propagation
        hidden_v = zeros(size(wv_i,1),size(input,2));
        for j = 1:size(wv_i,1)
            sum1_v = 0;
            for k = 1:length(wv_i)
                sum1_v = sum1_v  + wv_i(j,k)*input(k);
            end
            hidden_v(j,1) = tanh(sum1_v);
        end
        sum2_v =0;
        mat = vertcat(hidden_v,ones(1,1));
        for j = 1:length(wv_o)
            sum2_v = sum2_v + wv_o(j)*mat(j);
        end
        out_v(i) = exp(sum2_v); sig = out_v(i);
                        
       %% back propagation
        errors = Desired(i) - out_v(i);
        sse_v = sum(sum(errors.^2)); fprintf('sse_v = %f',sse_v);
        bouts = ((errors.*errors)./sig-1.0)/2;
        bps = zeros(length(wv_o),size(wv_o,1));
        for j = 1:size(wv_o,1)
            for k = 1:length(wv_o)
                bps(k,j) = wv_o(j,k)*bouts;
            end
        end
        bhs = zeros(size(hidden_v,1),size(hidden_v,2));
        for j = 1:size(hidden_v,1)
             bhs(j,:)= (1-hidden_v(j,:).^2).*bps(j,:);
        end     
        %% weight update for input to hidden network (Variance)
        for row = 1:size(bhs,1)
             for col = 1:size(input,1)
              wv_i(row,col) = wv_i(row,col)+learning_rate_v*bhs(row,:)*input(col,:); 
             end
        end          

	  %% weight update for hidden to output network (Variance)
        hidden_to_outs = [hidden_v;ones(1,1)]; 
        for j = 1:length(hidden_to_outs)
            wv_o(j) = wv_o(j)+learning_rate_v*bouts*hidden_to_outs(j);
        end
    end
end
prnout = out; 
plot([51:400],Desired(51:400),[51:400],prnout(51:400),[51:400],ph1_mean(51:400))
