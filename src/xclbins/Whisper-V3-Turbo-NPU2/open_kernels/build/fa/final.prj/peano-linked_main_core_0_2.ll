; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2p"

@s_tmp_f32_7_3 = external global [64 x float]
@r_tmp_7_3 = external global [64 x bfloat]
@s_tmp_7_3 = external global [64 x bfloat]
@sp_7_3 = external global [64 x float]
@up_7_3 = external global [64 x bfloat]
@gp_7_3 = external global [64 x [64 x bfloat]]
@g_7_3 = external global [64 x [64 x bfloat]]
@q_saved_7_3 = external global [64 x [64 x bfloat]]
@st_7_2 = external global [64 x float]
@rl_7_2 = external global [64 x bfloat]
@rc_7_2 = external global [64 x bfloat]
@up_s_7_2 = external global [64 x bfloat]
@sp_c_7_2 = external global [64 x float]
@up_c_7_2 = external global [64 x bfloat]
@gp_c_7_2 = external global [64 x [64 x bfloat]]
@s_tmp_f32_7_2 = external global [64 x float]
@r_tmp_7_2 = external global [64 x bfloat]
@s_tmp_7_2 = external global [64 x bfloat]
@sp_7_2 = external global [64 x float]
@up_7_2 = external global [64 x bfloat]
@gp_7_2 = external global [64 x [64 x bfloat]]
@g_7_2 = external global [64 x [64 x bfloat]]
@q_saved_7_2 = external global [64 x [64 x bfloat]]
@st_7_1 = external global [64 x float]
@rl_7_1 = external global [64 x bfloat]
@rc_7_1 = external global [64 x bfloat]
@up_s_7_1 = external global [64 x bfloat]
@sp_c_7_1 = external global [64 x float]
@up_c_7_1 = external global [64 x bfloat]
@gp_c_7_1 = external global [64 x [64 x bfloat]]
@s_tmp_f32_7_1 = external global [64 x float]
@r_tmp_7_1 = external global [64 x bfloat]
@s_tmp_7_1 = external global [64 x bfloat]
@sp_7_1 = external global [64 x float]
@up_7_1 = external global [64 x bfloat]
@gp_7_1 = external global [64 x [64 x bfloat]]
@g_7_1 = external global [64 x [64 x bfloat]]
@q_saved_7_1 = external global [64 x [64 x bfloat]]
@st_7_0 = external global [64 x float]
@rl_7_0 = external global [64 x bfloat]
@rc_7_0 = external global [64 x bfloat]
@up_s_7_0 = external global [64 x bfloat]
@sp_c_7_0 = external global [64 x float]
@up_c_7_0 = external global [64 x bfloat]
@gp_c_7_0 = external global [64 x [64 x bfloat]]
@s_tmp_f32_7_0 = external global [64 x float]
@r_tmp_7_0 = external global [64 x bfloat]
@s_tmp_7_0 = external global [64 x bfloat]
@sp_7_0 = external global [64 x float]
@up_7_0 = external global [64 x bfloat]
@gp_7_0 = external global [64 x [64 x bfloat]]
@g_7_0 = external global [64 x [64 x bfloat]]
@q_saved_7_0 = external global [64 x [64 x bfloat]]
@s_tmp_f32_6_3 = external global [64 x float]
@r_tmp_6_3 = external global [64 x bfloat]
@s_tmp_6_3 = external global [64 x bfloat]
@sp_6_3 = external global [64 x float]
@up_6_3 = external global [64 x bfloat]
@gp_6_3 = external global [64 x [64 x bfloat]]
@g_6_3 = external global [64 x [64 x bfloat]]
@q_saved_6_3 = external global [64 x [64 x bfloat]]
@st_6_2 = external global [64 x float]
@rl_6_2 = external global [64 x bfloat]
@rc_6_2 = external global [64 x bfloat]
@up_s_6_2 = external global [64 x bfloat]
@sp_c_6_2 = external global [64 x float]
@up_c_6_2 = external global [64 x bfloat]
@gp_c_6_2 = external global [64 x [64 x bfloat]]
@s_tmp_f32_6_2 = external global [64 x float]
@r_tmp_6_2 = external global [64 x bfloat]
@s_tmp_6_2 = external global [64 x bfloat]
@sp_6_2 = external global [64 x float]
@up_6_2 = external global [64 x bfloat]
@gp_6_2 = external global [64 x [64 x bfloat]]
@g_6_2 = external global [64 x [64 x bfloat]]
@q_saved_6_2 = external global [64 x [64 x bfloat]]
@st_6_1 = external global [64 x float]
@rl_6_1 = external global [64 x bfloat]
@rc_6_1 = external global [64 x bfloat]
@up_s_6_1 = external global [64 x bfloat]
@sp_c_6_1 = external global [64 x float]
@up_c_6_1 = external global [64 x bfloat]
@gp_c_6_1 = external global [64 x [64 x bfloat]]
@s_tmp_f32_6_1 = external global [64 x float]
@r_tmp_6_1 = external global [64 x bfloat]
@s_tmp_6_1 = external global [64 x bfloat]
@sp_6_1 = external global [64 x float]
@up_6_1 = external global [64 x bfloat]
@gp_6_1 = external global [64 x [64 x bfloat]]
@g_6_1 = external global [64 x [64 x bfloat]]
@q_saved_6_1 = external global [64 x [64 x bfloat]]
@st_6_0 = external global [64 x float]
@rl_6_0 = external global [64 x bfloat]
@rc_6_0 = external global [64 x bfloat]
@up_s_6_0 = external global [64 x bfloat]
@sp_c_6_0 = external global [64 x float]
@up_c_6_0 = external global [64 x bfloat]
@gp_c_6_0 = external global [64 x [64 x bfloat]]
@s_tmp_f32_6_0 = external global [64 x float]
@r_tmp_6_0 = external global [64 x bfloat]
@s_tmp_6_0 = external global [64 x bfloat]
@sp_6_0 = external global [64 x float]
@up_6_0 = external global [64 x bfloat]
@gp_6_0 = external global [64 x [64 x bfloat]]
@g_6_0 = external global [64 x [64 x bfloat]]
@q_saved_6_0 = external global [64 x [64 x bfloat]]
@s_tmp_f32_5_3 = external global [64 x float]
@r_tmp_5_3 = external global [64 x bfloat]
@s_tmp_5_3 = external global [64 x bfloat]
@sp_5_3 = external global [64 x float]
@up_5_3 = external global [64 x bfloat]
@gp_5_3 = external global [64 x [64 x bfloat]]
@g_5_3 = external global [64 x [64 x bfloat]]
@q_saved_5_3 = external global [64 x [64 x bfloat]]
@st_5_2 = external global [64 x float]
@rl_5_2 = external global [64 x bfloat]
@rc_5_2 = external global [64 x bfloat]
@up_s_5_2 = external global [64 x bfloat]
@sp_c_5_2 = external global [64 x float]
@up_c_5_2 = external global [64 x bfloat]
@gp_c_5_2 = external global [64 x [64 x bfloat]]
@s_tmp_f32_5_2 = external global [64 x float]
@r_tmp_5_2 = external global [64 x bfloat]
@s_tmp_5_2 = external global [64 x bfloat]
@sp_5_2 = external global [64 x float]
@up_5_2 = external global [64 x bfloat]
@gp_5_2 = external global [64 x [64 x bfloat]]
@g_5_2 = external global [64 x [64 x bfloat]]
@q_saved_5_2 = external global [64 x [64 x bfloat]]
@st_5_1 = external global [64 x float]
@rl_5_1 = external global [64 x bfloat]
@rc_5_1 = external global [64 x bfloat]
@up_s_5_1 = external global [64 x bfloat]
@sp_c_5_1 = external global [64 x float]
@up_c_5_1 = external global [64 x bfloat]
@gp_c_5_1 = external global [64 x [64 x bfloat]]
@s_tmp_f32_5_1 = external global [64 x float]
@r_tmp_5_1 = external global [64 x bfloat]
@s_tmp_5_1 = external global [64 x bfloat]
@sp_5_1 = external global [64 x float]
@up_5_1 = external global [64 x bfloat]
@gp_5_1 = external global [64 x [64 x bfloat]]
@g_5_1 = external global [64 x [64 x bfloat]]
@q_saved_5_1 = external global [64 x [64 x bfloat]]
@st_5_0 = external global [64 x float]
@rl_5_0 = external global [64 x bfloat]
@rc_5_0 = external global [64 x bfloat]
@up_s_5_0 = external global [64 x bfloat]
@sp_c_5_0 = external global [64 x float]
@up_c_5_0 = external global [64 x bfloat]
@gp_c_5_0 = external global [64 x [64 x bfloat]]
@s_tmp_f32_5_0 = external global [64 x float]
@r_tmp_5_0 = external global [64 x bfloat]
@s_tmp_5_0 = external global [64 x bfloat]
@sp_5_0 = external global [64 x float]
@up_5_0 = external global [64 x bfloat]
@gp_5_0 = external global [64 x [64 x bfloat]]
@g_5_0 = external global [64 x [64 x bfloat]]
@q_saved_5_0 = external global [64 x [64 x bfloat]]
@s_tmp_f32_4_3 = external global [64 x float]
@r_tmp_4_3 = external global [64 x bfloat]
@s_tmp_4_3 = external global [64 x bfloat]
@sp_4_3 = external global [64 x float]
@up_4_3 = external global [64 x bfloat]
@gp_4_3 = external global [64 x [64 x bfloat]]
@g_4_3 = external global [64 x [64 x bfloat]]
@q_saved_4_3 = external global [64 x [64 x bfloat]]
@st_4_2 = external global [64 x float]
@rl_4_2 = external global [64 x bfloat]
@rc_4_2 = external global [64 x bfloat]
@up_s_4_2 = external global [64 x bfloat]
@sp_c_4_2 = external global [64 x float]
@up_c_4_2 = external global [64 x bfloat]
@gp_c_4_2 = external global [64 x [64 x bfloat]]
@s_tmp_f32_4_2 = external global [64 x float]
@r_tmp_4_2 = external global [64 x bfloat]
@s_tmp_4_2 = external global [64 x bfloat]
@sp_4_2 = external global [64 x float]
@up_4_2 = external global [64 x bfloat]
@gp_4_2 = external global [64 x [64 x bfloat]]
@g_4_2 = external global [64 x [64 x bfloat]]
@q_saved_4_2 = external global [64 x [64 x bfloat]]
@st_4_1 = external global [64 x float]
@rl_4_1 = external global [64 x bfloat]
@rc_4_1 = external global [64 x bfloat]
@up_s_4_1 = external global [64 x bfloat]
@sp_c_4_1 = external global [64 x float]
@up_c_4_1 = external global [64 x bfloat]
@gp_c_4_1 = external global [64 x [64 x bfloat]]
@s_tmp_f32_4_1 = external global [64 x float]
@r_tmp_4_1 = external global [64 x bfloat]
@s_tmp_4_1 = external global [64 x bfloat]
@sp_4_1 = external global [64 x float]
@up_4_1 = external global [64 x bfloat]
@gp_4_1 = external global [64 x [64 x bfloat]]
@g_4_1 = external global [64 x [64 x bfloat]]
@q_saved_4_1 = external global [64 x [64 x bfloat]]
@st_4_0 = external global [64 x float]
@rl_4_0 = external global [64 x bfloat]
@rc_4_0 = external global [64 x bfloat]
@up_s_4_0 = external global [64 x bfloat]
@sp_c_4_0 = external global [64 x float]
@up_c_4_0 = external global [64 x bfloat]
@gp_c_4_0 = external global [64 x [64 x bfloat]]
@s_tmp_f32_4_0 = external global [64 x float]
@r_tmp_4_0 = external global [64 x bfloat]
@s_tmp_4_0 = external global [64 x bfloat]
@sp_4_0 = external global [64 x float]
@up_4_0 = external global [64 x bfloat]
@gp_4_0 = external global [64 x [64 x bfloat]]
@g_4_0 = external global [64 x [64 x bfloat]]
@q_saved_4_0 = external global [64 x [64 x bfloat]]
@s_tmp_f32_3_3 = external global [64 x float]
@r_tmp_3_3 = external global [64 x bfloat]
@s_tmp_3_3 = external global [64 x bfloat]
@sp_3_3 = external global [64 x float]
@up_3_3 = external global [64 x bfloat]
@gp_3_3 = external global [64 x [64 x bfloat]]
@g_3_3 = external global [64 x [64 x bfloat]]
@q_saved_3_3 = external global [64 x [64 x bfloat]]
@st_3_2 = external global [64 x float]
@rl_3_2 = external global [64 x bfloat]
@rc_3_2 = external global [64 x bfloat]
@up_s_3_2 = external global [64 x bfloat]
@sp_c_3_2 = external global [64 x float]
@up_c_3_2 = external global [64 x bfloat]
@gp_c_3_2 = external global [64 x [64 x bfloat]]
@s_tmp_f32_3_2 = external global [64 x float]
@r_tmp_3_2 = external global [64 x bfloat]
@s_tmp_3_2 = external global [64 x bfloat]
@sp_3_2 = external global [64 x float]
@up_3_2 = external global [64 x bfloat]
@gp_3_2 = external global [64 x [64 x bfloat]]
@g_3_2 = external global [64 x [64 x bfloat]]
@q_saved_3_2 = external global [64 x [64 x bfloat]]
@st_3_1 = external global [64 x float]
@rl_3_1 = external global [64 x bfloat]
@rc_3_1 = external global [64 x bfloat]
@up_s_3_1 = external global [64 x bfloat]
@sp_c_3_1 = external global [64 x float]
@up_c_3_1 = external global [64 x bfloat]
@gp_c_3_1 = external global [64 x [64 x bfloat]]
@s_tmp_f32_3_1 = external global [64 x float]
@r_tmp_3_1 = external global [64 x bfloat]
@s_tmp_3_1 = external global [64 x bfloat]
@sp_3_1 = external global [64 x float]
@up_3_1 = external global [64 x bfloat]
@gp_3_1 = external global [64 x [64 x bfloat]]
@g_3_1 = external global [64 x [64 x bfloat]]
@q_saved_3_1 = external global [64 x [64 x bfloat]]
@st_3_0 = external global [64 x float]
@rl_3_0 = external global [64 x bfloat]
@rc_3_0 = external global [64 x bfloat]
@up_s_3_0 = external global [64 x bfloat]
@sp_c_3_0 = external global [64 x float]
@up_c_3_0 = external global [64 x bfloat]
@gp_c_3_0 = external global [64 x [64 x bfloat]]
@s_tmp_f32_3_0 = external global [64 x float]
@r_tmp_3_0 = external global [64 x bfloat]
@s_tmp_3_0 = external global [64 x bfloat]
@sp_3_0 = external global [64 x float]
@up_3_0 = external global [64 x bfloat]
@gp_3_0 = external global [64 x [64 x bfloat]]
@g_3_0 = external global [64 x [64 x bfloat]]
@q_saved_3_0 = external global [64 x [64 x bfloat]]
@s_tmp_f32_2_3 = external global [64 x float]
@r_tmp_2_3 = external global [64 x bfloat]
@s_tmp_2_3 = external global [64 x bfloat]
@sp_2_3 = external global [64 x float]
@up_2_3 = external global [64 x bfloat]
@gp_2_3 = external global [64 x [64 x bfloat]]
@g_2_3 = external global [64 x [64 x bfloat]]
@q_saved_2_3 = external global [64 x [64 x bfloat]]
@st_2_2 = external global [64 x float]
@rl_2_2 = external global [64 x bfloat]
@rc_2_2 = external global [64 x bfloat]
@up_s_2_2 = external global [64 x bfloat]
@sp_c_2_2 = external global [64 x float]
@up_c_2_2 = external global [64 x bfloat]
@gp_c_2_2 = external global [64 x [64 x bfloat]]
@s_tmp_f32_2_2 = external global [64 x float]
@r_tmp_2_2 = external global [64 x bfloat]
@s_tmp_2_2 = external global [64 x bfloat]
@sp_2_2 = external global [64 x float]
@up_2_2 = external global [64 x bfloat]
@gp_2_2 = external global [64 x [64 x bfloat]]
@g_2_2 = external global [64 x [64 x bfloat]]
@q_saved_2_2 = external global [64 x [64 x bfloat]]
@st_2_1 = external global [64 x float]
@rl_2_1 = external global [64 x bfloat]
@rc_2_1 = external global [64 x bfloat]
@up_s_2_1 = external global [64 x bfloat]
@sp_c_2_1 = external global [64 x float]
@up_c_2_1 = external global [64 x bfloat]
@gp_c_2_1 = external global [64 x [64 x bfloat]]
@s_tmp_f32_2_1 = external global [64 x float]
@r_tmp_2_1 = external global [64 x bfloat]
@s_tmp_2_1 = external global [64 x bfloat]
@sp_2_1 = external global [64 x float]
@up_2_1 = external global [64 x bfloat]
@gp_2_1 = external global [64 x [64 x bfloat]]
@g_2_1 = external global [64 x [64 x bfloat]]
@q_saved_2_1 = external global [64 x [64 x bfloat]]
@st_2_0 = external global [64 x float]
@rl_2_0 = external global [64 x bfloat]
@rc_2_0 = external global [64 x bfloat]
@up_s_2_0 = external global [64 x bfloat]
@sp_c_2_0 = external global [64 x float]
@up_c_2_0 = external global [64 x bfloat]
@gp_c_2_0 = external global [64 x [64 x bfloat]]
@s_tmp_f32_2_0 = external global [64 x float]
@r_tmp_2_0 = external global [64 x bfloat]
@s_tmp_2_0 = external global [64 x bfloat]
@sp_2_0 = external global [64 x float]
@up_2_0 = external global [64 x bfloat]
@gp_2_0 = external global [64 x [64 x bfloat]]
@g_2_0 = external global [64 x [64 x bfloat]]
@q_saved_2_0 = external global [64 x [64 x bfloat]]
@s_tmp_f32_1_3 = external global [64 x float]
@r_tmp_1_3 = external global [64 x bfloat]
@s_tmp_1_3 = external global [64 x bfloat]
@sp_1_3 = external global [64 x float]
@up_1_3 = external global [64 x bfloat]
@gp_1_3 = external global [64 x [64 x bfloat]]
@g_1_3 = external global [64 x [64 x bfloat]]
@q_saved_1_3 = external global [64 x [64 x bfloat]]
@st_1_2 = external global [64 x float]
@rl_1_2 = external global [64 x bfloat]
@rc_1_2 = external global [64 x bfloat]
@up_s_1_2 = external global [64 x bfloat]
@sp_c_1_2 = external global [64 x float]
@up_c_1_2 = external global [64 x bfloat]
@gp_c_1_2 = external global [64 x [64 x bfloat]]
@s_tmp_f32_1_2 = external global [64 x float]
@r_tmp_1_2 = external global [64 x bfloat]
@s_tmp_1_2 = external global [64 x bfloat]
@sp_1_2 = external global [64 x float]
@up_1_2 = external global [64 x bfloat]
@gp_1_2 = external global [64 x [64 x bfloat]]
@g_1_2 = external global [64 x [64 x bfloat]]
@q_saved_1_2 = external global [64 x [64 x bfloat]]
@st_1_1 = external global [64 x float]
@rl_1_1 = external global [64 x bfloat]
@rc_1_1 = external global [64 x bfloat]
@up_s_1_1 = external global [64 x bfloat]
@sp_c_1_1 = external global [64 x float]
@up_c_1_1 = external global [64 x bfloat]
@gp_c_1_1 = external global [64 x [64 x bfloat]]
@s_tmp_f32_1_1 = external global [64 x float]
@r_tmp_1_1 = external global [64 x bfloat]
@s_tmp_1_1 = external global [64 x bfloat]
@sp_1_1 = external global [64 x float]
@up_1_1 = external global [64 x bfloat]
@gp_1_1 = external global [64 x [64 x bfloat]]
@g_1_1 = external global [64 x [64 x bfloat]]
@q_saved_1_1 = external global [64 x [64 x bfloat]]
@st_1_0 = external global [64 x float]
@rl_1_0 = external global [64 x bfloat]
@rc_1_0 = external global [64 x bfloat]
@up_s_1_0 = external global [64 x bfloat]
@sp_c_1_0 = external global [64 x float]
@up_c_1_0 = external global [64 x bfloat]
@gp_c_1_0 = external global [64 x [64 x bfloat]]
@s_tmp_f32_1_0 = external global [64 x float]
@r_tmp_1_0 = external global [64 x bfloat]
@s_tmp_1_0 = external global [64 x bfloat]
@sp_1_0 = external global [64 x float]
@up_1_0 = external global [64 x bfloat]
@gp_1_0 = external global [64 x [64 x bfloat]]
@g_1_0 = external global [64 x [64 x bfloat]]
@q_saved_1_0 = external global [64 x [64 x bfloat]]
@s_tmp_f32_0_3 = external global [64 x float]
@r_tmp_0_3 = external global [64 x bfloat]
@s_tmp_0_3 = external global [64 x bfloat]
@sp_0_3 = external global [64 x float]
@up_0_3 = external global [64 x bfloat]
@gp_0_3 = external global [64 x [64 x bfloat]]
@g_0_3 = external global [64 x [64 x bfloat]]
@q_saved_0_3 = external global [64 x [64 x bfloat]]
@st_0_2 = external global [64 x float]
@rl_0_2 = external global [64 x bfloat]
@rc_0_2 = external global [64 x bfloat]
@up_s_0_2 = external global [64 x bfloat]
@sp_c_0_2 = external global [64 x float]
@up_c_0_2 = external global [64 x bfloat]
@gp_c_0_2 = external global [64 x [64 x bfloat]]
@s_tmp_f32_0_2 = external global [64 x float]
@r_tmp_0_2 = external global [64 x bfloat]
@s_tmp_0_2 = external global [64 x bfloat]
@sp_0_2 = external global [64 x float]
@up_0_2 = external global [64 x bfloat]
@gp_0_2 = external global [64 x [64 x bfloat]]
@g_0_2 = external global [64 x [64 x bfloat]]
@q_saved_0_2 = external global [64 x [64 x bfloat]]
@st_0_1 = external global [64 x float]
@rl_0_1 = external global [64 x bfloat]
@rc_0_1 = external global [64 x bfloat]
@up_s_0_1 = external global [64 x bfloat]
@sp_c_0_1 = external global [64 x float]
@up_c_0_1 = external global [64 x bfloat]
@gp_c_0_1 = external global [64 x [64 x bfloat]]
@s_tmp_f32_0_1 = external global [64 x float]
@r_tmp_0_1 = external global [64 x bfloat]
@s_tmp_0_1 = external global [64 x bfloat]
@sp_0_1 = external global [64 x float]
@up_0_1 = external global [64 x bfloat]
@gp_0_1 = external global [64 x [64 x bfloat]]
@g_0_1 = external global [64 x [64 x bfloat]]
@q_saved_0_1 = external global [64 x [64 x bfloat]]
@st_0_0 = external global [64 x float]
@rl_0_0 = external global [64 x bfloat]
@rc_0_0 = external global [64 x bfloat]
@up_s_0_0 = external global [64 x bfloat]
@sp_c_0_0 = external global [64 x float]
@up_c_0_0 = external global [64 x bfloat]
@gp_c_0_0 = external global [64 x [64 x bfloat]]
@s_tmp_f32_0_0 = external global [64 x float]
@r_tmp_0_0 = external global [64 x bfloat]
@s_tmp_0_0 = external global [64 x bfloat]
@sp_0_0 = external global [64 x float]
@up_0_0 = external global [64 x bfloat]
@gp_0_0 = external global [64 x [64 x bfloat]]
@g_0_0 = external global [64 x [64 x bfloat]]
@q_saved_0_0 = external global [64 x [64 x bfloat]]
@GP_L1L2_0_buff_0 = external global [64 x [64 x bfloat]]
@GP_L1L2_0_cons_buff_0 = external global [64 x [64 x bfloat]]
@GP_L1L2_1_buff_0 = external global [64 x [64 x bfloat]]
@GP_L1L2_1_cons_buff_0 = external global [64 x [64 x bfloat]]
@GP_L1L2_2_buff_0 = external global [64 x [64 x bfloat]]
@GP_L1L2_2_cons_buff_0 = external global [64 x [64 x bfloat]]
@GP_L1L2_3_buff_0 = external global [64 x [64 x bfloat]]
@GP_L1L2_3_cons_buff_0 = external global [64 x [64 x bfloat]]
@GP_L1L2_4_buff_0 = external global [64 x [64 x bfloat]]
@GP_L1L2_4_cons_buff_0 = external global [64 x [64 x bfloat]]
@GP_L1L2_5_buff_0 = external global [64 x [64 x bfloat]]
@GP_L1L2_5_cons_buff_0 = external global [64 x [64 x bfloat]]
@GP_L1L2_6_buff_0 = external global [64 x [64 x bfloat]]
@GP_L1L2_6_cons_buff_0 = external global [64 x [64 x bfloat]]
@GP_L1L2_7_buff_0 = external global [64 x [64 x bfloat]]
@GP_L1L2_7_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L2L1_h0_s0_3_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L2L1_h0_s0_2_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L2L1_h0_s0_1_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L2L1_h0_s0_0_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L3L2_h0_s0_cons_buff_1 = external global [64 x [64 x bfloat]]
@QK_L3L2_h0_s0_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L2L1_h0_s1_3_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L2L1_h0_s1_2_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L2L1_h0_s1_1_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L2L1_h0_s1_0_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L3L2_h0_s1_cons_buff_1 = external global [64 x [64 x bfloat]]
@QK_L3L2_h0_s1_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L2L1_h0_s2_3_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L2L1_h0_s2_2_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L2L1_h0_s2_1_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L2L1_h0_s2_0_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L3L2_h0_s2_cons_buff_1 = external global [64 x [64 x bfloat]]
@QK_L3L2_h0_s2_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L2L1_h0_s3_3_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L2L1_h0_s3_2_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L2L1_h0_s3_1_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L2L1_h0_s3_0_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L3L2_h0_s3_cons_buff_1 = external global [64 x [64 x bfloat]]
@QK_L3L2_h0_s3_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L2L1_h1_s0_3_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L2L1_h1_s0_2_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L2L1_h1_s0_1_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L2L1_h1_s0_0_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L3L2_h1_s0_cons_buff_1 = external global [64 x [64 x bfloat]]
@QK_L3L2_h1_s0_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L2L1_h1_s1_3_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L2L1_h1_s1_2_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L2L1_h1_s1_1_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L2L1_h1_s1_0_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L3L2_h1_s1_cons_buff_1 = external global [64 x [64 x bfloat]]
@QK_L3L2_h1_s1_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L2L1_h1_s2_3_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L2L1_h1_s2_2_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L2L1_h1_s2_1_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L2L1_h1_s2_0_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L3L2_h1_s2_cons_buff_1 = external global [64 x [64 x bfloat]]
@QK_L3L2_h1_s2_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L2L1_h1_s3_3_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L2L1_h1_s3_2_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L2L1_h1_s3_1_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L2L1_h1_s3_0_cons_buff_0 = external global [64 x [64 x bfloat]]
@QK_L3L2_h1_s3_cons_buff_1 = external global [64 x [64 x bfloat]]
@QK_L3L2_h1_s3_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L2L1_h0_s0_3_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L2L1_h0_s0_2_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L2L1_h0_s0_1_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L2L1_h0_s0_0_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L3L2_h0_s0_cons_buff_1 = external global [64 x [64 x bfloat]]
@V_L3L2_h0_s0_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L2L1_h0_s1_3_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L2L1_h0_s1_2_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L2L1_h0_s1_1_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L2L1_h0_s1_0_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L3L2_h0_s1_cons_buff_1 = external global [64 x [64 x bfloat]]
@V_L3L2_h0_s1_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L2L1_h0_s2_3_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L2L1_h0_s2_2_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L2L1_h0_s2_1_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L2L1_h0_s2_0_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L3L2_h0_s2_cons_buff_1 = external global [64 x [64 x bfloat]]
@V_L3L2_h0_s2_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L2L1_h0_s3_3_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L2L1_h0_s3_2_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L2L1_h0_s3_1_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L2L1_h0_s3_0_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L3L2_h0_s3_cons_buff_1 = external global [64 x [64 x bfloat]]
@V_L3L2_h0_s3_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L2L1_h1_s0_3_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L2L1_h1_s0_2_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L2L1_h1_s0_1_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L2L1_h1_s0_0_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L3L2_h1_s0_cons_buff_1 = external global [64 x [64 x bfloat]]
@V_L3L2_h1_s0_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L2L1_h1_s1_3_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L2L1_h1_s1_2_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L2L1_h1_s1_1_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L2L1_h1_s1_0_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L3L2_h1_s1_cons_buff_1 = external global [64 x [64 x bfloat]]
@V_L3L2_h1_s1_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L2L1_h1_s2_3_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L2L1_h1_s2_2_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L2L1_h1_s2_1_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L2L1_h1_s2_0_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L3L2_h1_s2_cons_buff_1 = external global [64 x [64 x bfloat]]
@V_L3L2_h1_s2_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L2L1_h1_s3_3_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L2L1_h1_s3_2_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L2L1_h1_s3_1_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L2L1_h1_s3_0_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L3L2_h1_s3_cons_buff_1 = external global [64 x [64 x bfloat]]
@V_L3L2_h1_s3_cons_buff_0 = external global [64 x [64 x bfloat]]

declare void @debug_i32(i32)

; Unknown intrinsic
declare void @llvm.aie2p.event(i32)

; Unknown intrinsic
declare void @llvm.aie2p.put.ms(i32, i32)

; Unknown intrinsic
declare { i32, i32 } @llvm.aie2p.get.ss()

; Unknown intrinsic
declare void @llvm.aie2p.mcd.write.vec(<16 x i32>, i32)

; Unknown intrinsic
declare <16 x i32> @llvm.aie2p.scd.read.vec(i32)

; Unknown intrinsic
declare void @llvm.aie2p.acquire(i32, i32)

; Unknown intrinsic
declare void @llvm.aie2p.release(i32, i32)

; Unknown intrinsic
declare void @llvm.aie2p.set.ctrl.reg(i32, i32)

declare void @copy_tile(ptr, ptr)

declare void @matmul_a_b_bf16(ptr, ptr, ptr)

declare void @matmul_g_b_bf16(ptr, ptr, ptr)

declare void @zero_fill_gp_bf16(ptr)

declare void @zero_fill_g_bf16(ptr)

declare void @neg_inf_fill_up_bf16(ptr)

declare void @maximum_up_u_bf16(ptr, ptr)

declare void @exp_up_minus_u(ptr, ptr, ptr)

declare void @mul_r_gp(ptr, ptr)

declare void @add_gp_g(ptr, ptr)

declare void @vector_copy_32elems(i32, ptr, ptr)

declare void @apply_length_mask(ptr, i32)

declare void @fused_softmax(ptr, ptr, ptr, ptr)

declare void @zero_fill_sp_f32(ptr)

declare void @widen_bf16_to_f32(ptr, ptr)

declare void @vector_copy_32elems_f32(i32, ptr, ptr)

declare void @accum_sp_r_s_f32(ptr, ptr, ptr)

declare void @div_gp_sp_f32(ptr, ptr)

declare void @cascade_get3(ptr, ptr, ptr)

declare void @cascade_put3(ptr, ptr, ptr)

define void @core_0_2() {
  br label %1

1:                                                ; preds = %21, %0
  %2 = phi i64 [ %22, %21 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775807
  br i1 %3, label %4, label %23

4:                                                ; preds = %19, %1
  %5 = phi i64 [ %20, %19 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 10
  br i1 %6, label %7, label %21

7:                                                ; preds = %17, %4
  %8 = phi i64 [ %18, %17 ], [ 0, %4 ]
  %9 = icmp slt i64 %8, 6
  br i1 %9, label %10, label %19

10:                                               ; preds = %7
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @copy_tile(ptr @QK_L2L1_h0_s0_0_cons_buff_0, ptr @q_saved_0_0)
  call void @llvm.aie2p.release(i32 50, i32 1)
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.aie2p.release(i32 50, i32 1)
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.aie2p.release(i32 50, i32 1)
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.aie2p.release(i32 50, i32 1)
  call void @zero_fill_gp_bf16(ptr @gp_0_0)
  call void @zero_fill_sp_f32(ptr @sp_0_0)
  call void @neg_inf_fill_up_bf16(ptr @up_0_0)
  br label %11

11:                                               ; preds = %14, %10
  %12 = phi i64 [ %16, %14 ], [ 0, %10 ]
  %13 = icmp slt i64 %12, 6
  br i1 %13, label %14, label %17

14:                                               ; preds = %11
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @zero_fill_g_bf16(ptr @g_0_0)
  call void @matmul_a_b_bf16(ptr @q_saved_0_0, ptr @QK_L2L1_h0_s0_0_cons_buff_0, ptr @g_0_0)
  %15 = trunc i64 %12 to i32
  call void @apply_length_mask(ptr @g_0_0, i32 %15)
  call void @fused_softmax(ptr @g_0_0, ptr @up_0_0, ptr @s_tmp_0_0, ptr @r_tmp_0_0)
  call void @mul_r_gp(ptr @r_tmp_0_0, ptr @gp_0_0)
  call void @matmul_g_b_bf16(ptr @g_0_0, ptr @V_L2L1_h0_s0_0_cons_buff_0, ptr @gp_0_0)
  call void @widen_bf16_to_f32(ptr @s_tmp_0_0, ptr @s_tmp_f32_0_0)
  call void @accum_sp_r_s_f32(ptr @sp_0_0, ptr @r_tmp_0_0, ptr @s_tmp_f32_0_0)
  call void @vector_copy_32elems_f32(i32 0, ptr @s_tmp_f32_0_0, ptr @sp_0_0)
  call void @llvm.aie2p.release(i32 50, i32 1)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %16 = add i64 %12, 1
  br label %11

17:                                               ; preds = %11
  call void @cascade_get3(ptr @gp_c_0_0, ptr @up_c_0_0, ptr @sp_c_0_0)
  call void @vector_copy_32elems(i32 0, ptr @up_0_0, ptr @up_s_0_0)
  call void @maximum_up_u_bf16(ptr @up_c_0_0, ptr @up_0_0)
  call void @exp_up_minus_u(ptr @up_c_0_0, ptr @up_0_0, ptr @rc_0_0)
  call void @exp_up_minus_u(ptr @up_s_0_0, ptr @up_0_0, ptr @rl_0_0)
  call void @mul_r_gp(ptr @rc_0_0, ptr @gp_c_0_0)
  call void @mul_r_gp(ptr @rl_0_0, ptr @gp_0_0)
  call void @add_gp_g(ptr @gp_0_0, ptr @gp_c_0_0)
  call void @zero_fill_sp_f32(ptr @st_0_0)
  call void @accum_sp_r_s_f32(ptr @sp_c_0_0, ptr @rc_0_0, ptr @st_0_0)
  call void @accum_sp_r_s_f32(ptr @sp_0_0, ptr @rl_0_0, ptr @st_0_0)
  call void @vector_copy_32elems_f32(i32 0, ptr @st_0_0, ptr @sp_c_0_0)
  call void @div_gp_sp_f32(ptr @sp_c_0_0, ptr @gp_c_0_0)
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  call void @copy_tile(ptr @gp_c_0_0, ptr @GP_L1L2_0_buff_0)
  call void @llvm.aie2p.release(i32 53, i32 1)
  %18 = add i64 %8, 1
  br label %7

19:                                               ; preds = %7
  %20 = add i64 %5, 1
  br label %4

21:                                               ; preds = %4
  %22 = add i64 %2, 1
  br label %1

23:                                               ; preds = %1
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
