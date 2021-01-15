function [v,f,recon,edge_index,neighbour,vdiff] = meshinfo(mesh)
firstobj=mesh;
[ v, f, ~, ~, ~, VVsimp, CotWeight,~,~,~,edge_index] = cotlp(firstobj);
edge_index=[edge_index;edge_index(:,[2,1])]';
% edge_index = full(CotWeight);
W = full(CotWeight);
W(W>0)=1;
W(W<0)=-1;
pointnum = size(W,1);
D = zeros(pointnum);
for i = 1:pointnum
    D(i,i) = sum(W(i,:));
end
L = D-W;
recon = (L'*L)\L';

% 
neighbour=zeros(size(v,1),100);
maxnum=0;
for i=1:size(VVsimp,1)
    neighbour(i,1:size(VVsimp{i,:},2))=VVsimp{i,:};
    if size(VVsimp{i,:},2)>maxnum
        maxnum=size(VVsimp{i,:},2);
    end
end
neighbour(:,maxnum+1:end)=[];
% 
% v = readobjfromfile('\\10.41.0.202\yangjie\yangjie\structurenet\code\tmpglobal.obj');
for i = 1:pointnum
    for j = 1:maxnum
        curneighbour = neighbour(i,j);
        if curneighbour == 0
            break
        end
        w = W(i,curneighbour)/2;
        vdiff(i,j,:) = w*(v(i,:)-v(curneighbour,:));
        edges1Ring(i,j) = sqrt(sum((v(i,:)-v(curneighbour,:)).^2));
        if neighbour(i,j)>0
            %                 cotweight(i,j)=CotWeight(i,neighbour2(i,j));
            cotweight(i,j)=1/length(nonzeros(neighbour(i,:)));
        end
    end
end
% W1 = full(L_);
% for i = 1:size(W1,1)
%     for j = 1:size(W1,2)
%         if W1(i,j) ~= 0
%             W1(i,j)=1;
%         end
%     end
% end
end