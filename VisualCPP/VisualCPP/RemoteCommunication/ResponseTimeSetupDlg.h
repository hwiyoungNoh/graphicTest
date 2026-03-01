#pragma once
#include "afxwin.h"
#include "afxcmn.h"

#include "CRColorimeter.h"

// CResponseTimeSetupDlg dialog

class CResponseTimeSetupDlg : public CDialogEx
{
	DECLARE_DYNAMIC(CResponseTimeSetupDlg)

public:
	CResponseTimeSetupDlg(CWnd* pParent = NULL);   // standard constructor
	virtual ~CResponseTimeSetupDlg();

// Dialog Data
	enum { IDD = IDD_RESPONSETIMESETUP };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

	DECLARE_MESSAGE_MAP()
	BOOL OnInitDialog();
public:
	void SetColorimeter(CCRColorimeter* pColorimeter);
	void EnableSettings(BOOL enable);
	void UpdateSettings(BOOL enable);
public:
	// Response Time mode
	int m_nMode;
	// Peak index selector
	int m_nPeaks;
	int m_nFilterType;
	int m_nAverage;
	BOOL m_bEnableClippingLimits;
	float m_fClippingLowerLimit;
	float m_fClippingUpperLimit;
	float m_fStepZoneLowerLimit;
	float m_fStepZoneUpperLimit;
	CComboBox m_filterTypeComboBox;
	CSpinButtonCtrl m_clippingLoSpinButtonCtrl;
	CSpinButtonCtrl m_clippingHiSpinButtonCtrl;
	CSpinButtonCtrl m_stepZoneLoSpinButtonCtrl;
	CSpinButtonCtrl m_stepZoneHiSpinButtonCtrl;
	CButton m_modeAutoRadioButtonCtrl;
	CButton m_modeManualRadioButtonCtrl;

	CButton m_enableClippingCheckBox;
	
	CEdit m_peaksEdit;
	CEdit m_averageEdit;
	CEdit m_clippingLoEdit;
	CEdit m_clippingHiEdit;
	CEdit m_zoneLoEdit;
	CEdit m_zoneHiEdit;
	CEdit m_stepZoneLoEdit;
	CEdit m_stepZoneHiEdit;
	CCRColorimeter* m_pColorimeter;
	
};
