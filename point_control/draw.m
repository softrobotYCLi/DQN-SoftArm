%%µ√µΩŒÛ≤Óæ‡¿Î_dis
dis = [];
a = 1;
for i = 1:1000
    dis(a,1) = abs(poscur(i,1) - postarget(i,1)) + abs(poscur(i,2) - postarget(i,2));
    a = a + 1;
end
%% get error fig1
a = 1;
for i = 795:891
    err1(a,1) = abs(poscur(i,1) - postarget(i,1));
    err1(a,2) = abs(poscur(i,2) - postarget(i,2));
    a = a + 1;
end
a = 1;
for i = 892:1029
    err2(a,1) = abs(poscur(i,1) - postarget(i,1));
    err2(a,2) = abs(poscur(i,2) - postarget(i,2));
    a = a + 1;
end
a = 1;
for i = 1848:2156
    err3(a,1) = abs(poscur(i,1) - postarget(i,1));
    err3(a,2) = abs(poscur(i,2) - postarget(i,2));
    a = a + 1;
end

%% draw πÏº£
subplot(3,2,[1 3 5])
plot(pos1(:,1),pos1(:,2));
hold on;
plot(pos2(:,1),pos2(:,2));
hold on;
plot(pos3(:,1),pos3(:,2));

%% ª≠Õº
subplot(3,2,2)
plot(line1);
hold on;
plot(line2);
hold on;
plot(line3);
hold on;
subplot(3,2,4)
plot(err1(:,1));
hold on;
plot(err2(:,1));
hold on;
plot(err3(:,1));
hold on;
subplot(3,2,6)
plot(err1(:,2));
hold on;
plot(err2(:,2));
hold on;
plot(err3(:,2));
hold on;
