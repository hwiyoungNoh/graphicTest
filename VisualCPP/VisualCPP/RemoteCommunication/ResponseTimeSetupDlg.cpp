// ResponseTimeSetupDlg.cpp : implementation file
//

#include "stdafx.h"

#include "RemoteCommunication.h"
#include "ResponseTimeSetupDlg.h"

#include <CRColorimeter.h>


// CResponseTimeSetupDlg dialog

IMPLEMENT_DYNAMIC(CResponseTimeSetupDlg, CDialogEx)

CResponseTimeSetupDlg::CResponseTimeSetupDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(CResponseTimeSetupDlg::IDD, pParent)
	, m_nMode(0)
	, m_nPeaks(0)
	, m_nFilterType(0)
	, m_nAverage(__CR_FILTER_MOVING_AVERAGE_DEFAULT)
	, m_bEnableClippingLimits(FALSE)
	, m_fClippingLowerLimit(__CR_CLIPPING_LO_DEFAULT * 100.0F)
	, m_fClippingUpperLimit(__CR_CLIPPING_HI_DEFAULT * 100.0F)
	, m_fStepZoneLowerLimit(__CR_STEPZONE_LO_DEFAULT * 100.0F)
	, m_fStepZoneUpperLimit(__CR_STEPZONE_HI_DEFAULT * 100.0F)
{
	
	m_pColorimeter = NULL;
}

CResponseTimeSetupDlg::~CResponseTimeSetupDlg()
{
}

void CResponseTimeSetupDlg::SetColorimeter(CCRColorimeter* pColorimeter)
{
	m_pColorimeter = pColorimeter;
}

void CResponseTimeSetupDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_EDIT_PEAKS, m_nPeaks);
	DDV_MinMaxInt(pDX, m_nPeaks, 0, 100);
	DDX_CBIndex(pDX, IDC_COMBO_FILTER_TYPE, m_nFilterType);
	DDX_Text(pDX, IDC_EDIT_AVERAGE, m_nAverage);
	//DDX_Radio(pDX, IDC_RADIO_AUTO, m_nMode);
	//DDX_Radio(pDX, IDC_RADIO_MANUAL, m_nMode);
	DDV_MinMaxInt(pDX, m_nAverage, __CR_FILTER_MOVING_AVERAGE_MIN, __CR_FILTER_MOVING_AVERAGE_MAX);
	DDX_Check(pDX, IDC_CHECK_CLIPPING_LIMITS, m_bEnableClippingLimits);
	DDX_Text(pDX, IDC_EDIT_CLIPPING_LOWER, m_fClippingLowerLimit);
	DDX_Text(pDX, IDC_EDIT_CLIPPING_UPPER, m_fClippingUpperLimit);
	DDV_MinMaxFloat(pDX, m_fClippingLowerLimit, __CR_CLIPPING_MIN * 100.0F, __CR_CLIPPING_MAX * 100.0F);
	DDV_MinMaxFloat(pDX, m_fClippingUpperLimit, __CR_CLIPPING_MIN * 100.0F, __CR_CLIPPING_MAX * 100.0F);
	DDX_Text(pDX, IDC_EDIT_STEPZONE_LOWER, m_fStepZoneLowerLimit);
	DDX_Text(pDX, IDC_EDIT_STEPZONE_UPPER, m_fStepZoneUpperLimit);
	DDV_MinMaxFloat(pDX, m_fStepZoneLowerLimit, __CR_STEPZONE_MIN * 100.0F, __CR_STEPZONE_MAX * 100.0F);
	DDV_MinMaxFloat(pDX, m_fStepZoneUpperLimit, __CR_STEPZONE_MIN * 100.0F, __CR_STEPZONE_MAX * 100.0F);
	DDX_Control(pDX, IDC_COMBO_FILTER_TYPE, m_filterTypeComboBox);
	DDX_Control(pDX, IDC_SPIN_CLIPPING_LOWER, m_clippingLoSpinButtonCtrl);
	DDX_Control(pDX, IDC_SPIN_CLIPPING_UPPER, m_clippingHiSpinButtonCtrl);
	DDX_Control(pDX, IDC_SPIN_STEPZONE_LOWER, m_stepZoneLoSpinButtonCtrl);
	DDX_Control(pDX, IDC_SPIN_STEPZONE_UPPER, m_stepZoneHiSpinButtonCtrl);
	
	DDX_Control(pDX, IDC_EDIT_AVERAGE, m_averageEdit);
	DDX_Control(pDX, IDC_EDIT_PEAKS, m_peaksEdit);
	DDX_Control(pDX, IDC_RADIO_AUTO, m_modeAutoRadioButtonCtrl);
	DDX_Control(pDX, IDC_RADIO_MANUAL, m_modeManualRadioButtonCtrl);
	DDX_Control(pDX, IDC_CHECK_CLIPPING_LIMITS, m_enableClippingCheckBox);
	DDX_Control(pDX, IDC_EDIT_CLIPPING_UPPER, m_clippingHiEdit);
	DDX_Control(pDX, IDC_EDIT_CLIPPING_LOWER, m_clippingLoEdit);
	DDX_Control(pDX, IDC_EDIT_STEPZONE_UPPER, m_stepZoneHiEdit);
	DDX_Control(pDX, IDC_EDIT_STEPZONE_LOWER, m_stepZoneLoEdit);
	

}


BEGIN_MESSAGE_MAP(CResponseTimeSetupDlg, CDialogEx)
END_MESSAGE_MAP()


// CResponseTimeSetupDlg message handlers

BOOL CResponseTimeSetupDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();
	// Extra initialization here
	
		
	m_filterTypeComboBox.AddString(_T("None"));
	m_filterTypeComboBox.AddString(_T("Moving Window Average"));

	m_nFilterType = 0;
	m_nPeaks = 0;
	
	m_nAverage = __CR_FILTER_MOVING_AVERAGE_DEFAULT;

	m_bEnableClippingLimits = FALSE;
	m_fClippingLowerLimit = __CR_CLIPPING_LO_DEFAULT * 100.0F;
	m_fClippingUpperLimit = __CR_CLIPPING_HI_DEFAULT * 100.0F;
	m_fStepZoneLowerLimit = __CR_STEPZONE_LO_DEFAULT * 100.0F;
	m_fStepZoneUpperLimit = __CR_STEPZONE_HI_DEFAULT * 100.0F;

	m_clippingLoSpinButtonCtrl.SetRange32((int)__CR_CLIPPING_MIN * 100.0F, (int)__CR_CLIPPING_MAX * 100.0F);
	m_clippingHiSpinButtonCtrl.SetRange32((int)__CR_CLIPPING_MIN * 100.0F, (int)__CR_CLIPPING_MAX * 100.0F);
	m_stepZoneLoSpinButtonCtrl.SetRange32((int)__CR_STEPZONE_MIN * 100.0F, (int)__CR_STEPZONE_MAX * 100.0F);
	m_stepZoneHiSpinButtonCtrl.SetRange32((int)__CR_STEPZONE_MIN * 100.0F, (int)__CR_STEPZONE_MAX * 100.0F);
	

	UpdateData(FALSE);

	return TRUE;
}



void CResponseTimeSetupDlg::EnableSettings(BOOL enable)
{
    m_modeAutoRadioButtonCtrl.EnableWindow(enable);
    m_modeManualRadioButtonCtrl.EnableWindow(enable);
   	
	m_peaksEdit.EnableWindow(enable);

	m_filterTypeComboBox.EnableWindow(enable);
	m_averageEdit.EnableWindow(enable);
  
	m_enableClippingCheckBox.EnableWindow(enable);

    m_clippingLoEdit.EnableWindow(enable);
    m_clippingHiEdit.EnableWindow(enable);
    m_stepZoneLoEdit.EnableWindow(enable);
    m_stepZoneHiEdit.EnableWindow(enable);
    
}


void CResponseTimeSetupDlg::UpdateSettings(BOOL enable)
{   
    if (m_pColorimeter)
    {
      
		/*CString strValue;
		strValue.Format(_T("%f"), m_pColorimeter->FlickerMaxSearchFrequency());
		m_pDataView->m_responseTimeSetupCtrl.m_maxFreqFlickerSearchEdit.SetWindowText(strValue);
		m_pDataView->m_responseTimeSetupCtrl.m_filterTypeComboBox.SetCurSel(m_pColorimeter->FlickerFilterType());
		strValue.Format(_T("%d"), m_pColorimeter->FlickerFilterOrder());
		m_pDataView->m_responseTimeSetupCtrl.m_orderEdit.SetWindowText(strValue);
		strValue.Format(_T("%f"), m_pColorimeter->FlickerFilterFrequency());
		m_pDataView->m_responseTimeSetupCtrl.m_frequencyEdit.SetWindowText(strValue);
		strValue.Format(_T("%f"), m_pColorimeter->FlickerFilterBandwidth());
		m_pDataView->m_responseTimeSetupCtrl.m_bandwidthEdit.SetWindowText(strValue);*/

    }
}