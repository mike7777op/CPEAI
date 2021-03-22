

def score1(pred1,pred2,pred3):
    for j,k,l in zip(pred1,pred2,pred3):
        # top_inds1 = j.argsort()[::-1][:5]
        # top_inds2 = k.argsort()[::-1][:5]
        # top_inds3 = l.argsort()[::-1][:5]
        # s1 = top_inds1[0]
        # s2 = top_inds2[0]
        # s3 = top_inds3[0]
        s1 = j
        s2 = k 
        s3 = l
        if s1>s2:
            if s2>s3:
                # print('s1>s2>s3')
                return 's1'
            elif s2 == s3:
                # print('s1>s2=s3')
                return 's1'
            elif s1>s2 and s2<s3:
                if s1<s3:
                    # print('s3>s1>s2')
                    return 's3'
                elif s1 == s3:
                    # print('s3=s1>2')
                    return 's3=s1'
                elif s1>s3:
                    # print('s1>s3>s2')
                    return 's1'
        else:
            if s1 == s2:
                if s2>s3:
                    # print('s1=s2>s3')
                    return 's1=s2'
                elif s2 == s3:
                    # print('s1=s2=s3')
                    return 's1=s2=s3'
                else:
                    if s1<s3:
                        # print('s3>s1=s2')
                        return 's3'
                    elif s1==s3:
                        # print('s1=s2=s3')
                        return 's1=s2=s3'
                    elif s1>s3:
                        # print('s1=s2>s3')
                        return 's1=s2'
            elif s1<s2:
                if s2<s3:
                    # print('s3>s2>s1')
                    return 's3'
                else:
                    if s2 == s3:
                        # print('s2=s3>s1')
                        return 's2=s3'
                    else:
                        if s1<s3:
                            # print('s2>s3>s1')
                            return 's2'
                        elif s1 == s3:
                            # print('s2>s3>s1')
                            return 's2'
                        elif s1>s3:
                            # print('s2>s1>s3')
                            return 's2'

pred1 = input()
pred2 = input()
pred3 = input()

print(score1(pred1,pred2,pred3))