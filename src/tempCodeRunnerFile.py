def residual(theta_U):
            UBJ_curr=upper(theta_U,(input_corner_dict['UBJ']-input_corner_dict['UA']),input_corner_dict['UA'],corner_k_U)
            LBJ_curr=lower(UBJ_curr,(input_corner_dict['LBJ']-input_corner_dict['LA']),input_corner_dict['LA'],corner_k_L,seed1,const_d1)[0]
            TRO_curr=tierod(UBJ_curr,LBJ_curr,input_corner_dict['UBJ'],input_corner_dict['LBJ'],input_corner_dict['TRO'],seed2,input_corner_dict['TRI'],const_d2)[0]
            WC_curr=triad_transform(UBJ_curr,LBJ_curr,TRO_curr,input_corner_dict['UBJ'],input_corner_dict['LBJ'],input_corner_dict['TRO'],input_corner_dict['WC'])
            roll_radius=np.linalg.norm(np.array([WC_curr[1],WC_curr[2]]),res)
            centre_travel=roll_radius*np.sin(np.radians(alpha))
            return (WC_curr[2]-input_corner_dict['WC'][2]-centre_travel)
        
        root3=fsolve(residual,seed3)[0]